"""FINETUNING ON TARGET TASK: CLASSIFICATION"""

"""
Author: Leonardo Antiqui <leonardoantiqui@gmail.com>

"""

"""Libraries"""

import os
import argparse
import subprocess
import requests

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

get_code_by_name(drive, 'training_functions.py')


"""Importing from other files"""

#Downloading functions for classification and contrastive learning

from training_functions import *


"""File paths"""

experiment_path = ''
os.makedirs(experiment_path + 'Models/', exist_ok=True)

"""Training algorithm"""

def main_finetuning(DATASET, MODEL_NAME, WEIGHTS=None, LEARNING_RATE=None, OPTIMIZER_NAME='SGD', WEIGHT_DECAY=None, scheduler=None, MOMENTUM=0.9,
         IMG_SIZE=None, BATCH_SIZE=None, N_EPOCHS=None, CLIP_V=None):
         
    """
    
    Fine-tuning on Multiclass classification task
    
    Args:
        DATASET (str): name of target dataset
        MODEL_NAME (str): CNN type
        WEIGHTS (str or dict): initialization weights. Default None
        LEARNING_RATE (float): learning rate. Default None
        OPTIMIZER_NAME (str): Optimizer type. Default 'SGD'
        WEIGHT_DECAY (float): weight decay for L2 regularization. Default None
        scheduler (str): type learning rate schedule. Default None
        MOMENTUM (float): momentum. Default 0.9
        IMG_SIZE (int): image size of the synthetic images. Default None
        BATCH_SIZE (int): batch size. Default None
        CLIP_V (float): gradient clipping. Default None
        
    Returns:
        max_test_acc (float): maximum test accuracy achieved during traning
        auc (float): area under the ROC Curve
        convergence_speed (float): convergence speed
    
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    DEVICE = torch.device(device)

    # Authenticate with Google Drive using Service Account authentication
    gauth = GoogleAuth()
    scope = ["https://www.googleapis.com/auth/drive"]
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secrets.json', scope)

    drive = GoogleDrive(gauth)

    if IMG_SIZE is None:
        IMG_SIZE = configs[DATASET]['img_size']
    if BATCH_SIZE is None:
        BATCH_SIZE = configs[DATASET]['batch_size']
    SUBSET_SIZE = configs[DATASET]['subset']

    train_loader, test_loader = dataloader(DATASET, IMG_SIZE, BATCH_SIZE, SUBSET_SIZE)

    print('Train dataset size:', len(train_loader.dataset))
    print('Test dataset size:', len(test_loader.dataset))


    N_CLASSES = configs[DATASET]['n_classes']

    if LEARNING_RATE is None:
        LEARNING_RATE = configs[DATASET][MODEL_NAME]['learning_rate']

    if WEIGHT_DECAY is None:
        WEIGHT_DECAY = configs[DATASET][MODEL_NAME]['weight_decay']

    if N_EPOCHS is None:
        N_EPOCHS = configs[DATASET][MODEL_NAME]['n_epochs']

    PRINT_F = 2
    print('Model:', WEIGHTS)

    if WEIGHTS == None or WEIGHTS == 'IMAGENET1K_V1':
        encoder = create_model(MODEL_NAME, weights=WEIGHTS)
    else:
        get_model_by_name(drive, WEIGHTS)

        PATH = experiment_path + 'Models/' + WEIGHTS
        WEIGHTS = torch.load(PATH, map_location='cpu')

        encoder = create_model(MODEL_NAME, weights=WEIGHTS)
        encoder.load_state_dict(WEIGHTS, strict=False)

    #Freezing model parameters
    # for param in encoder.parameters():
    #     param.requires_grad = False

    #Reinitializing model parameters
    # param_reinitialization(encoder, n=2)

    encoder = head_change(encoder, N_CLASSES)
    encoder.to(DEVICE)

    loss = torch.nn.CrossEntropyLoss() #cross entropy loss between input logits and target

    #Optimizer
    extra = [MOMENTUM, #momentum
            WEIGHT_DECAY, #weight decay
            False #Nesterov
            ]

    opt = create_optimizer(encoder.parameters(), OPTIMIZER_NAME, LEARNING_RATE, extra)

    if scheduler:

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=1)

    print(f'MODEL: {encoder.name}, ',
          f'DATASET: {DATASET}, ',
          f'IMG_SIZE: {IMG_SIZE}, ',
          f'N_EPOCHS: {N_EPOCHS}, ',
          f'N_CLASSES: {N_CLASSES}, ',
          f'OPT: {OPTIMIZER_NAME}, ',
          f'LEARNING_RATE: {LEARNING_RATE}, ',
          f'WEIGHT_DECAY: {WEIGHT_DECAY}, ',
          f'BATCH_SIZE: {BATCH_SIZE} ',
          f'DATE-TIME: {time.strftime("%m.%d-%H%M")}'
          )

    log_dir = experiment_path + "Logs/"

    filename_suffix = '.' + encoder.name + '_' + DATASET + '_' + time.strftime("%Y%m%d_%H%M")
    print(filename_suffix)

    writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)

    writer.add_graph(encoder, torch.randn((BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)).to(device))

    best_model_wts = fit_classf(encoder, loss, opt, train_loader, test_loader, N_EPOCHS, scheduler, writer, clip_value=CLIP_V, device=DEVICE, print_f=PRINT_F)

    writer.close()

    encoder.load_state_dict(best_model_wts)

    file_paths = [file for file in os.listdir(experiment_path + 'Logs') if file.split(".")[-1] == filename_suffix[1:]]



    #Extracting max test accuracy
    data = extract_scalars(experiment_path + 'Logs/' + file_paths[0])
    test_accs = data['test_accuracy']['value']
    max_test_acc = round(max(test_accs),3)
    print('Max test accuracy:', max_test_acc)

    #Top5 accuracy
    k = 5
    topk_acc = accuracy_classf(encoder, test_loader,  k=k, device=DEVICE)
    print(f'top{k} accuracy: {round(topk_acc, 3)}')

    #AUC
    auc = auc_classf(encoder, test_loader, num_classes=N_CLASSES, device=DEVICE)
    print(f'AUC: {round(auc, 3)}')

    #Speed convergence
    reference_file = configs[DATASET][MODEL_NAME]['ref_file']
    get_log_by_name(drive, reference_file)
    convergence_speed = converge_diff(filename_suffix[1:], reference_file, experiment_path)
    print(f'convergence: {convergence_speed}')

    #Saving the model
    if max_test_acc > configs[DATASET][MODEL_NAME]['max_top1']*1.1:
        model_file_name = f'Model_{encoder.name}_trainedOn{DATASET}_{time.strftime("%Y%m%d_%H%M")}.pth'
        torch.save(encoder.state_dict(), experiment_path + 'Models/' + model_file_name)
        print('Finetuned model:', model_file_name)
    #    #Uploading Model in Google drive
    #    upload_model_by_name(drive, model_file_name)

    #Uploading log in Google drive
    upload_log_by_name(drive, file_paths[0])

    return max_test_acc, auc, convergence_speed
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Finetuning')
    
    #Method attaches individual argument specifications to the parser
    parser.add_argument("--ds", type=str, help="Specify dataset name")
    parser.add_argument("--net", type=str, help="Specify DNN name")
    parser.add_argument("--weights", type=str, help="Specify weights to use in DNN")
    parser.add_argument("--lr", type=float, help="Specify learning rate")
    parser.add_argument("--opt", type=str, default='SGD', help="Specify optimizer name")
    parser.add_argument("--wd", type=float, help="Specify L2 weight decay")
    parser.add_argument("--scheduler", type=str, help="Specify the scheduler")
    parser.add_argument("--momentum", type=float, default=0.9, help="Specify momentum")
    parser.add_argument("--imgsize", type=int, help="Specify image size")
    parser.add_argument("--bs", type=int, help="Specify batch size")
    parser.add_argument("--epochs", type=int, help="Specify intial number of epochs per task")
    parser.add_argument("--clipv", type=float, help="Specify clipping value")
    
    args = parser.parse_args() #method runs the parser and places the extracted data in a argparse.Namespace object
    
    main_finetuning(
        args.ds,
        args.net,
        args.weights,
        args.lr,
        args.opt,
        args.wd,
        args.scheduler,
        args.momentum,
        args.imgsize,
        args.bs,
        args.epochs,
        args.clipv,
    )