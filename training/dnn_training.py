import os
import torch as t
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import uuid
import optuna
import pandas

from earlystopping import EarlyStopping
from net import Net

# Constant value
MAX_EPOCH = 1000
PATIENCE  = 20

# Training
def train_step(epochs, data_loader, model, optimizer):

    running_loss = 0.0
    check_interval = 10

    model.train()

    for i, (input, label) in enumerate(data_loader) :

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
def val_step(model, data_loader, early_stopping):

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

        early_stopping(correct/total, model)
        print('------------------------------------------')
        # print('Validation loss: %f' %(running_loss/itr))

        # Accを返す
        return correct/total

def objective(trial):

    uuid_ = str(uuid.uuid4())
    out_dir = 'output/'+uuid_
    os.makedirs(out_dir)

    trial.set_user_attr('uuid', uuid_)

    out_units = 2

    # ハイパーパラメータ
    batch_size   = trial.suggest_int('batch_size', 32, 512)
    dropout      = trial.suggest_float('dropout', 0.1, 0.5)
    lr           = trial.suggest_float('lr', 1e-5, 1e-2)
    mid_c        = trial.suggest_int('mid_c', 8, 16, 2)
    out_c        = trial.suggest_int('out_c', 18, 32, 2)
    hidden_units = trial.suggest_int('hidden_units', 1024, 4096)
    out_units    = trial.suggest_int('out_units', 128, 256)

    # devフォルダにある画像を取り込み
    train_datasets = ImageFolder(root='dataset/dev/train', transform=data_transform['train'])
    val_datasets   = ImageFolder(root='dataset/dev/val',   transform=data_transform['val'])

    # DataLoader作成
    train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(val_datasets,   batch_size=batch_size, shuffle=False, num_workers=4)

    # データサイズを取得
    in_c, h, w = train_datasets[0][0].shape

    model = Net(in_c, h, w, mid_c, out_c, hidden_units, out_units, dropout).to(device)

    optimizer = t.optim.SGD(model.parameters(), lr)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, out_dir=out_dir, key='max')

    acc = 0

    # DNN training -------------------------------------------------------
    for epoch in range(MAX_EPOCH+1):

        train_step(epoch, train_dataloader, model, optimizer)
        acc = val_step(model, val_dataloader, early_stopping)

        trial.report(acc, epoch+1)

        # Earlystopping
        if early_stopping.early_stop:
            print('Early stopping.')
            break
        
        # Pruner
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    # --------------------------------------------------------------------

    return acc

if __name__ == "__main__":

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')  # CUDA or CPU
    print('Device :', device)

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

    # Loss関数と最適化法の定義
    criterion = nn.CrossEntropyLoss().to(device)

    # 枝刈り手法
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=4,
        min_early_stopping_rate=0
    )

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100)

    study_df = study.trials_dataframe()
    study_df.to_csv("result.csv")