import os
from datetime import date
import torch as t
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from earlystopping import EarlyStopping
from net import Net

MAX_EPOCH = 1000
PATIENCE  = 20
BATCH_SIZE   = 32
LEARNING_RATE = 0.0001

# Training
def train_step(epochs, data_loader, model, optimizer):

    running_loss = 0.0
    check_interval = 30

    model.train()

    for i, (input, label) in enumerate(data_loader) :

        input.requires_grad_()

        input = input.to(device)
        label = label.to(device).long()

        # initialize optimizer
        optimizer.zero_grad()

        output = model(input)

        # loss計算
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 定期的に Training loss を表示&ログに追加
        if i % check_interval == check_interval-1:
            print('[%d, %3d] loss: %f' %(epochs+1, i+1, running_loss/check_interval))
            running_loss = 0

# Validation
def val_step(model, data_loader, out_dir=None):

    running_loss = 0.0
    correct = 0
    total = 0
    itr = 0

    model.eval()
    with t.no_grad():
        for input, label in data_loader:
            
            input = input.to(device)
            label = label.to(device).long()

            output = model(input)
            
            _, predicted = t.max(output.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

            loss = criterion(output, label)
            running_loss += loss.item()

            itr += 1

        print('correct: {:d}  total: {:d}'.format(correct, total))
        print('accuracy = {:f}'.format(correct / total))

        early_stopping(1 - (correct/total), model, out_dir)
        print('------------------------------------------')
        print('Validation loss: %f' %(running_loss/itr))

if __name__ == "__main__":

    # Transforms
    data_transform = {
        'train': transforms.Compose([
                 transforms.RandomResizedCrop(225, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]),
                 ]),
        'val': transforms.Compose([
                 transforms.RandomResizedCrop(225, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]),
                 ]),
    }
    
    # devフォルダにある画像を取り込み
    train_datasets = ImageFolder(root='dataset/dev/train', transform=data_transform['train'])
    val_datasets   = ImageFolder(root='dataset/dev/val',   transform=data_transform['val'])

    # DataLoader作成
    train_dataloader = DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True )
    val_dataloader   = DataLoader(val_datasets,   batch_size=BATCH_SIZE, shuffle=False)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')  # CUDA or CPU
    print('Device :', device)

    # 結果を出力するフォルダ
    out_dir = 'output/'+date.today().strftime('%Y%m%d')

    # モデル保存用フォルダ作成（YYYYMMDD_ID）
    for i in range(1, 100):
        _out_dir = out_dir+'_'+str(i).rjust(2,'0')
        tmp = os.path.isdir(_out_dir)
        if not os.path.isdir(_out_dir):
            os.makedirs(out_dir+'_'+str(i).rjust(2,'0'))
            out_dir = _out_dir
            break
    
    model = Net().to(device)

    # Loss関数と最適化法の定義
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = t.optim.SGD(model.parameters(), LEARNING_RATE)

    early_stopping = EarlyStopping(PATIENCE, verbose=False)

    #####################    DNN Training     #########################
    for epoch in range(MAX_EPOCH+1):

        train_step(epoch, train_dataloader, model, optimizer)
        val_step(model, val_dataloader, out_dir=out_dir)

        if early_stopping.early_stop:
            print('Early stopping.')
            break