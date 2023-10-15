from dataset import *
from UT_HAR_model import *
from NTU_Fi_model import *
from widar_model import *
from self_supervised_model import *
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split
from util import load_data_n_model
import time as t
import psutil
import subprocess

def is_gpu_available():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,nounits,noheader'])
        return True
    except FileNotFoundError:
        return False
# Function to get GPU memory usage using nvidia-smi
def get_gpu_memory():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        memory_used = [int(x) // 1048576 for x in result.decode('utf-8').strip().split('\n')]  # Convert bytes to MB
        print(f"GPU Memory Used: {memory_used}")
        return memory_used
    except FileNotFoundError:
        return []

def train(model, train_loader, val_loader, num_epochs, learning_rate, criterion, device,patience=7):
    
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_test_acc = 0
    best_val_acc = 0
    no_improvement_count = 0  # Count of epochs with no improvement
    best_model_state = None
    # Get the initial memory usage
    ram_memory_before = psutil.virtual_memory().used // 1048576  # Convert bytes to MB
    gpu_memory_before = get_gpu_memory() if is_gpu_available() else []

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    inference_time = []
    for epoch in range(num_epochs):
        # Training phase

        model.train()
        start_time = t.time()
        epoch_loss = 0
        epoch_accuracy = 0
        
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        

        epoch_loss = epoch_loss / len(train_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(train_loader)
        end_time = t.time()
        epoch_inference_time = end_time - start_time
        inference_time.append(epoch_inference_time)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.type(torch.LongTensor)
                
                outputs = model(inputs)
                outputs = outputs.to(device)
                outputs = outputs.type(torch.FloatTensor)
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                predict_y = torch.argmax(outputs, dim=1).to(device)
                val_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_accuracy / len(val_loader)
        # Print training and validation metrics
        print(f'Epoch: {epoch + 1}, Train Accuracy: {epoch_accuracy:.4f}, Train Loss: {epoch_loss:.9f}, Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.9f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_accuracy
            best_test_loss = epoch_loss
            best_test_acc =  epoch_accuracy
            no_improvement_count = 0
            
            # Save the best model state
            best_model_state = model.state_dict()
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f'Validation loss did not improve for {patience} epochs. Early stopping.')
            break  # Exit the training loop

      # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
        
        
        
        
    #  saving all data in a file
    with open('Information.txt', 'a') as file:
        file.write(f"Number of Epochs: {num_epochs}\n")
        file.write(f"Learning Rate: {learning_rate}\n")
        file.write(f"=== Each epoch time ===\n")
        for epoch, time in enumerate(inference_time):
          file.write(f"\tEpoch {epoch + 1}:\t{time:.2f} seconds\n")
    # Get the memory usage after running the training function
    # Get the memory usage after running the training function
    ram_memory_after = psutil.virtual_memory().used // 1048576  # Convert bytes to MB
    gpu_memory_after = get_gpu_memory() if is_gpu_available() else []

    # Calculate the memory used by your function
    ram_memory_used = ram_memory_after - ram_memory_before
    gpu_memory_used = [gpu_after - gpu_before for gpu_before, gpu_after in zip(gpu_memory_before, gpu_memory_after)]

    print(f"RAM Memory Used: {ram_memory_used} MB")
    print(f"GPU Memory Used: {gpu_memory_used}")
    with open('Information.txt', 'a') as file:
        file.write(f"Training accuracy: {best_test_acc:.4f}, Training loss: {best_test_loss:.5f}\n")
        file.write(f"Validation accuracy: {best_val_acc:.4f}, Validation loss: {best_val_loss:.5f}\n")
        file.write(f"RAM Memory Used: {ram_memory_used} MB\n")
    if gpu_memory_used:
        with open('Information.txt', 'a') as file:
            file.write(f"GPU Memory Used: {gpu_memory_used} MB\n")

    return


def test(model,tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    inference_time = []
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)

    num_test_rows = len(tensor_loader.dataset) * tensor_loader.batch_size
    num_test_features = tensor_loader.dataset[0][0].shape[1]

    with open('Information.txt', 'a') as file:
        file.write("validation accuracy:{:.4f}, loss:{:.5f}\n".format(float(test_acc),float(test_loss)))

    print("Testing accuracy:{:.4f},  Testing loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return

    
def main():
    root = './Data/' 
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    args = parser.parse_args()

    train_loader, val_loader,test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = t.time()
    num_train_rows = len(train_loader.dataset) * train_loader.batch_size
    num_train_features = train_loader.dataset[0][0].shape[1]
    print("Number of Rows in Training Data: ", num_train_rows)
    print("Number of Features in Training Data: ", num_train_features)
    
    with open('Information.txt', 'a') as file:
      file.write(f"Model: {args.model}, Dataset: {args.dataset})\n")
      file.write(f"-------------------------------------------------------------------------\n")
      file.write(f"-------------------------------------------------------------------------\n")
      file.write(f"====== FOR TRAINING ======\n")
      file.write(f"Number of Rows in Training Data: {num_train_rows}\n")
      file.write(f"Number of Features in Training Data: {num_train_features}\n")
        

    train(
        model=model,
        train_loader= train_loader,
        val_loader=val_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device
         )

    end_time = t.time()

    inference_time = end_time - start_time
 

    with open('Information.txt', 'a') as file:
      file.write(f" Inference Time: {inference_time:.2f} seconds\n")
      
    print("Inference time: {:.2f} seconds".format(inference_time))

    
    with open('Information.txt', 'a') as file:
      file.write(f"====== FOR Testing ======\n")
      file.write(f"Number of Rows in Testing Data: {num_train_rows}\n")
      file.write(f"Number of Features in testing Data: {num_train_features}\n")
    

    start_time = t.time()
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device
        )
    end_time = t.time()
    inference_time = end_time - start_time
    with open('Information.txt', 'a') as file:
      file.write(f" Inference Time: {inference_time:.2f} seconds\n")
      file.write(f"-----------------------END-----------------------\n")
      file.write(f"\n")

    return


if __name__ == "__main__":
    main()
