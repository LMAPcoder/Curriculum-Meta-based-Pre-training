"""CONTRASTIVE-PRETRAINING"""

"""
Author: Leonardo Antiqui <leonardoantiqui@gmail.com>

"""

"""Libraries"""

import os
import argparse
import subprocess
import requests

# Authenticate with Google Drive using Service Account authentication
#if os.system("pip show pydrive2") != 0:
#    os.system("pip install --quiet pydrive2")

try:
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
except:
    subprocess.check_call(["pip", "install", "--quiet", "pydrive2"])
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    from oauth2client.service_account import ServiceAccountCredentials
    
"""Acquiring python files"""

gauth = GoogleAuth()
scope = ["https://www.googleapis.com/auth/drive"]
gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secrets.json', scope)

drive = GoogleDrive(gauth)

def check_internet_connection():
    try:
        # Attempt to make a GET request to a known website
        response = requests.get("http://www.google.com", timeout=5)
        # Check if the response status code indicates success (2xx)
        if response.status_code == 200:
            return True
        else:
            return False
    except:
        return False
        
def get_code_by_name(drive, title):
    if check_internet_connection():
        file_path = title
        if not os.path.exists(file_path):
            foldered_list=drive.ListFile({'q': "'1UHTbF3RmzTjmoEGQ_4dcGRPFe1qtnAZC' in parents and trashed=false"}).GetList()
            for file in foldered_list:
                if(file['title']==title):
                    model = drive.CreateFile({'id': file['id']})
                    model.GetContentFile(file_path,
                                        # remove_bom=False
                                        )
                    print(f'{title} donwloaded!')
                
#Downloading auxiliary files

get_code_by_name(drive, 'generator.py')
get_code_by_name(drive, 'meta_teacher.py')
get_code_by_name(drive, 'training_functions.py')


"""Importing modules from python files"""

#Downloading the functions to create RGB images with PIL

from generator import *

#Downloading functions to create triplets according to curricula
from meta_teacher import *

#Downloading functions for classification and contrastive learning

from training_functions import *

"""Loading curricula"""

def load_curricula(filename, experiment_path):

    """
    
    Convert a txt file into a dictionary containing the curriculum
    
    Args:
        filename (str): name of the txt file containing the curriculum
        experiment_path (str): path of the folder containing the experiment results
    
    Returns:
        curricula (dict): dictionary containing the curriculum
    
    """

    curricula_file = experiment_path + 'Curricula/' + filename
    with open(curricula_file, 'r') as file:
        lines = file.readlines()

    curricula = dict()
    task = None
    for line in lines:
        if line[:5] == 'Tasks':
            if task:
                curricula[task]['specs'] = specs
            task = line.strip()
            curricula[task] = dict()
            flag = False
            specs = list()
            continue
        elif line[:6] == 'lesson':
            curricula[task]['lesson'] = eval(line.strip()[8:])
        elif line[:5] == 'specs':
            flag = True
            continue
        if flag:
            elements = line[1:-2].split(', ')
            spec = list(map(int, elements))
            specs.append(spec)

    curricula[task]['specs'] = specs

    return curricula
    
    
"""File paths"""
experiment_path = ''
os.makedirs(experiment_path + 'Models/', exist_ok=True)
os.makedirs(experiment_path + 'Curricula/', exist_ok=True)

"""Training algorithm"""

def main_pretraining(curricula_name, MODEL_NAME, EMB_DIM=100, LOSS='Norm2', LEARNING_RATE=0.001, OPTIMIZER_NAME='SGD', WEIGHT_DECAY=0.001, SCHEDULER=None,
                     L1_LAMBDA=None, MOMENTUM=0.9, MARGIN=1.0, Diff_LR=False, IMG_SIZE=100, NUM_examples=200, NUM_task_replay=2, BATCH_SIZE=20,
                     N_EPOCHS=1, EXTRA_nt=None, CLIP_V=None, MIN_acc=None, ONE_SHOT=False, BEST_model=False):
                     
    """
    
    Curriculum Meta-based Pretraining function with Synthetic Data
    
    Args:
        curricula_name (str): name of curriculum file
        MODEL_NAME (str): CNN type
        EMB_DIM (int): embedding dimensionality. Default 100
        LOSS (str): type of contrastive loss and distance metric. Default 'Norm2'
        LEARNING_RATE (float): learning rate. Default 0.001
        OPTIMIZER_NAME (str): Optimizer type. Default 'SGD'
        WEIGHT_DECAY (float): weight decay for L2 regularization. Default 0.001
        SCHEDULER (str): type learning rate schedule. Default None
        L1_LAMBDA (float): weight decay for L1 regularization. Default None
        MOMENTUM (float): momentum. Default 0.9
        MARGIN (float): margin of triplet loss. Default 1.0
        Diff_LR (bool): mode eneabling differencial rate. Default False
        IMG_SIZE (int): image size of the synthetic images. Default 100
        NUM_examples (int). number of examples per pre-training task. Default 200
        NUM_task_replay (int): number of simulataneusly learned tasks. Current tasks + previous tasks. Default 2
        BATCH_SIZE (int): batch size. Default 20
        N_EPOCHS (int): number of epochs per pre-training task. Default 1
        EXTRA_nt (int): number of pre-training tasks before increasing number of epochs and training examples per task. Default None
        CLIP_V (float): gradient clipping. Default None
        MIN_acc (float): minimal validation accuracy required before moving to subsequent pre-training task. Default None
        ONE_SHOT (bool): mode eneabling the creation of new examples per training epohc within a task. Default  False
        BEST_mode (bool): mode eneabling the transfer of the best parameters weights to the subsequent pre-training task. Default False
        
    Returns:
        model_file_name (str): name of the pre-trained model
    
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    DEVICE = torch.device(device)

    # Authenticate with Google Drive using Service Account authentication
    gauth = GoogleAuth()
    scope = ["https://www.googleapis.com/auth/drive"]
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secrets.json', scope)

    drive = GoogleDrive(gauth)

    #Curricula

    get_curricula_by_name(drive, curricula_name)

    curricula = load_curricula(curricula_name, experiment_path)

    print(curricula_name)

    #Model
    encoder = create_model(MODEL_NAME, EMB_DIM)

    state_dict = encoder.state_dict() #to save the original state of the model
    state_dict_original_encoder = copy.deepcopy(state_dict)

    encoder.to(DEVICE)

    #Triplet loss
    loss = loss_func(LOSS, MARGIN)

    #Optimizer
    extra = [MOMENTUM, #momentum
            WEIGHT_DECAY, #weight decay
            False #Nesterov
            ]

    if Diff_LR:
        parameters = list()
        n_layers = sum(1 for _ in encoder.parameters())
        for i, (name, param) in enumerate(encoder.named_parameters()):
            if 'fc' in name or 'classifier' in name:
                parameters.append({'params': param, 'lr': LEARNING_RATE})
            else:
                parameters.append({'params': param, 'lr': max(0.0001, (LEARNING_RATE/n_layers)*(i+1))})
    else:
        parameters = encoder.parameters()

    opt = create_optimizer(parameters, OPTIMIZER_NAME, LEARNING_RATE, extra)

    #Scheduler
    scheduler = SCHEDULER
    if SCHEDULER:
        if SCHEDULER == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3, eta_min=0.0001)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

    #Training

    image_size = (IMG_SIZE, IMG_SIZE)

    PRINT_eF = 1
    PRINT_tF = 3

    EXTRA_ep = 1 #add epochs
    EXTRA_ex = 50 #add examples

    MAX_rep = 3 #Maximum repetition times

    DATASET = 'Dummies'
    #[shape type, shape count, shape color, shape pattern, background color, background pattern]
    # DIST_weights = [1.0,0.8,0.6,0.3,0.4,0.3]
    # DIFF_weight = 0.08

    # set_setting('distance', DIST_weights)
    # set_setting('difficulty', DIFF_weight)

    print(f'MODEL: {encoder.name}, ',
          f'DATASET: {DATASET}, ',
          f'LOSS: {LOSS}, ',
          f'MARGIN: {MARGIN}, ',
          f'IMG_SIZE: {IMG_SIZE}, ',
          f'N_EPOCHS: {N_EPOCHS}(+{EXTRA_ep}%{EXTRA_nt}), ',
          f'EMB_DIM: {EMB_DIM}, ',
          f'OPT: {OPTIMIZER_NAME}, ',
          f'LEARNING_RATE: {LEARNING_RATE}, ',
          f'SCHEDULER: {SCHEDULER}, ',
          f'MOMENTUM: {MOMENTUM}, ',
          f'N_EXAMPLES: {NUM_examples}(+{EXTRA_ex}%{EXTRA_nt}), ',
          f'BATCH_SIZE: {BATCH_SIZE}, ',
          f'REPLAY: {NUM_task_replay}, ',
          f'CLIP: {CLIP_V}, ',
          f'WEIGHT_DECAY: {WEIGHT_DECAY}, ',
          f'L1_REG: {L1_LAMBDA}, ',
          f'Diff_LR: {Diff_LR}, ',
          f'MIN_acc: {MIN_acc}, ',
          f'BEST_model: {BEST_model}, ',
          f'ONE_shot: {ONE_SHOT}, ',
          f'DATE-TIME: {time.strftime("%Y%m%d-%H%M")}'
          )

    log_dir = experiment_path + "Logs/"

    filename_suffix = '.' + encoder.name + '_' + DATASET + '_' + time.strftime("%Y%m%d_%H%M")
    print(filename_suffix)

    writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)

    writer.add_graph(encoder, torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)).to(device))

    #Task progression
    for task in curricula:

        lesson = curricula[task]['lesson']
        specs =  curricula[task]['specs']

        epochs = N_EPOCHS
        n_examples = NUM_examples
        batch_s = BATCH_SIZE

        n_specs = len(specs)
        # n_specs = 30
        print(lesson, n_specs)

        #Generation on-the-fly
        # test_dl_dummy_pool = torch.utils.data.DataLoader(range(NUM_examples_pool), batch_size=batch_s,
        #                                                       collate_fn=lambda batch: custom_collate(batch, lesson, specs, image_size),
        #                                                       num_workers=2)

        train_dl_dummy_pool = torch.utils.data.DataLoader(triplet_dataset(500, lesson, specs, image_size),
                                                        batch_size=batch_s, num_workers=2)
        test_dl_dummy_pool = torch.utils.data.DataLoader(triplet_dataset(100, lesson, specs, image_size),
                                                        batch_size=batch_s, num_workers=2)

        start_time = time.time()

        #Example progression
        i = 0
        rep = 0
        while i < n_specs:

            k = i
            # k = random.randint(0, len(specs))
            print(specs[k], 'Task:', i, ' N_epochs:', epochs, ' N_examples:', n_examples)

            writer.add_scalar(tag='N_epochs', scalar_value=epochs, global_step=i)
            writer.add_scalar(tag='N_examples', scalar_value=n_examples, global_step=i)

            #Sequence of specifications
            spec = specs[max(0, k - NUM_task_replay):k+1] #Replay

            #Examples generated on the fly
            if ONE_SHOT:

                #Generating train and test dummy data loaders on-the-fly for one-shot learning strategy
                train_dl_dummy = torch.utils.data.DataLoader(range(n_examples), batch_size=batch_s,
                                                            collate_fn=lambda batch: custom_collate(batch, lesson, spec, image_size),
                                                            num_workers=2)
            else:

                train_dl_dummy = torch.utils.data.DataLoader(triplet_dataset(n_examples, lesson, spec, image_size),
                                                            batch_size=batch_s, num_workers=2)

            #Validation dataset
            test_dl_dummy = torch.utils.data.DataLoader(triplet_dataset(100, lesson, spec, image_size),
                                                            batch_size=batch_s, num_workers=2)

            #Training
            best_model_wts = fit_contrast(encoder, loss, opt, train_dl_dummy, test_dl_dummy,
                                          epochs, scheduler, writer, l1_lambda=L1_LAMBDA, clip_value=CLIP_V, device=DEVICE, print_f=PRINT_eF)


            #Keeping best model
            if BEST_model:
                encoder.load_state_dict(best_model_wts) #To load the best parameters of the task

            #Minimum test accuracy required
            if MIN_acc:
                if accuracy_contrast(encoder, test_dl_dummy, DEVICE) < MIN_acc:
                    rep += 1
                    if rep < MAX_rep:
                        continue

            #Repetitions
            i += 1
            rep = 0

            #Additinal epochs and examples
            if EXTRA_nt:
                if i%EXTRA_nt==0:
                    epochs += EXTRA_ep
                    n_examples += EXTRA_ex

            #Printing and recording global test accuracy
            if i%PRINT_tF == 0:

                # test_acc_pool = accuracy_contrast(encoder, test_dl_dummy_pool, DEVICE)
                test_acc_pool = test_acc_pool_func(encoder, MODEL_NAME, EMB_DIM, train_dl_dummy_pool,
                                                   test_dl_dummy_pool, LOSS, EPOCHS=3, device=DEVICE)

                writer.add_scalar(tag='test_acc_pool', scalar_value=test_acc_pool, global_step=i)

                print(f'test_acc_pool: {test_acc_pool:.2f}')

        
        #Final test accuracy measurement
        # test_acc_pool = accuracy_contrast(encoder, test_dl_dummy_pool, DEVICE)
        test_acc_pool = test_acc_pool_func(encoder, MODEL_NAME, EMB_DIM, train_dl_dummy_pool,
                                           test_dl_dummy_pool, LOSS, EPOCHS=3, device=DEVICE)

        writer.add_scalar(tag='test_acc_pool', scalar_value=test_acc_pool, global_step=i)

        print(f'test_acc_pool: {test_acc_pool:.2f}')
        
        end_time = time.time()
        time_train = end_time - start_time
        print(f'Total training time: {time_train/60:.2f}')

    writer.close()

    #Uploading log in Google drive
    file_paths = [file for file in os.listdir(experiment_path + 'Logs') if file.split(".")[-1] == filename_suffix[1:]]
    upload_log_by_name(drive, file_paths[0])

    #Saving the model
    model_file_name = f'Model_{encoder.name}_trainedOn{DATASET}_{time.strftime("%Y%m%d_%H%M")}.pth'
    os.makedirs(experiment_path + 'Models/', exist_ok=True)
    torch.save(encoder.state_dict(), experiment_path + 'Models/' + model_file_name)
    print('Pretrained model:', model_file_name)

    #Uploading Model in Google drive
    upload_model_by_name(drive, model_file_name)


    return model_file_name
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='DummyTraining')
    
    #Method attaches individual argument specifications to the parser
    parser.add_argument("--curricula", type=str, help="Specify curricula name file")
    parser.add_argument("--net", type=str, help="Specify DNN name")
    parser.add_argument("--dim", type=int, default=100, help="Specify embeddding dimensionality")
    parser.add_argument("--loss", type=str, default='Norm2', help="Specify the loss function")
    parser.add_argument("--lr", type=float, default=0.001, help="Specify learning rate")
    parser.add_argument("--opt", type=str, default='SGD', help="Specify optimizer name")
    parser.add_argument("--wd", type=float, default=0.001, help="Specify L2 weight decay")
    parser.add_argument("--scheduler", type=str, help="Specify the scheduler")
    parser.add_argument("--l1", type=float, help="Specify L1 weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Specify momentum")
    parser.add_argument("--mrg", type=float, default=1.0, help="Specify margin")
    parser.add_argument("--difflr", type=bool, default=False, help="Specify differencial learning rate")
    parser.add_argument("--imgsize", type=int, default=100, help="Specify image size")
    parser.add_argument("--numexs", type=int, default=200, help="Specify initial number of examples per task")
    parser.add_argument("--replay", type=int, default=2, help="Specify number of tasks in replay")
    parser.add_argument("--bs", type=int, default=20, help="Specify batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Specify intial number of epochs per task")
    parser.add_argument("--extasks", type=int, help="Specify number of tasks before increasing epochs and examples")
    parser.add_argument("--clipv", type=float, help="Specify clipping value")
    parser.add_argument("--minacc", type=float, help="Specify minimum accuracy per task")
    parser.add_argument("--oneshot", type=bool, default=False, help="Specify one shot regime")
    parser.add_argument("--best", type=bool, default=False, help="Specify passing best model per task")
    
    args = parser.parse_args() #method runs the parser and places the extracted data in a argparse.Namespace object
    
    main_pretraining(
        args.curricula,
        args.net,
        args.dim,
        args.loss,
        args.lr,
        args.opt,
        args.wd,
        args.scheduler,
        args.l1,
        args.momentum,
        args.mrg,
        args.difflr,
        args.imgsize,
        args.numexs,
        args.replay,
        args.bs,
        args.epochs,
        args.extasks,
        args.clipv,
        args.minacc,
        args.oneshot,
        args.best,
    )