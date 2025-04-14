import matplotlib.pyplot as plt 
import re 
import os 


# Regex patterns for extracting data
train_loss_pattern = re.compile(r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\].*?Loss ([\d.]+)")
eval_metric_pattern = re.compile(r"lexicon0: ([\w]*): ([\d.]+)")

def get_curves_from_logfile(log_file:str): 
    train_losses = [] # (epoch, train_loss)
    current_epoch_losses = [] 
    metrics = [] # (epoch progress, metric)

    curr_full_epochs = 0 
    curr_epoch_progress = 0.0 

    with open(log_file, "r") as file:
        for line in file:
            # Extract training loss
            train_match = train_loss_pattern.search(line)
            if train_match:
                #print("TRAIN MATCH:", train_match)
                epoch = int(train_match.group(1))
                curr_epoch_progress = float(train_match.group(2)) / float(train_match.group(3)) 
                loss = float(train_match.group(4))

                # check if full epoch done 
                if epoch > curr_full_epochs: 
                    # Calculate and store the average loss for the current epoch
                    if current_epoch_losses:
                        avg_loss = sum(current_epoch_losses) / len(current_epoch_losses)
                        train_losses.append((epoch, avg_loss)) 
                        current_epoch_losses = []
                    curr_full_epochs = epoch

                # Add batch loss to current epoch's losses
                current_epoch_losses.append(loss)

            # Extract evaluation metric
            eval_match = eval_metric_pattern.search(line)
            if eval_match:
                metric_name = eval_match.group(1)
                metric = float(eval_match.group(2))
                metrics.append((curr_full_epochs+curr_epoch_progress, metric)) 
    
    return train_losses, metrics, metric_name 

                


startdir = './logs'

_, subdirs, _ = next(os.walk('./logs')) 

for subdir in subdirs:

    train_losses, metrics, metric_name = get_curves_from_logfile(os.path.join(startdir, subdir, 'log.txt'))

    if metric_name == 'accuracy': 
        metric_name = 'Accuracy' 
    elif metric_name == 'editdistance': 
        metric_name = 'Edit Distance' 
    else: 
        print("UNKNOWN METRIC NAME:", metric_name)

    fig, ax1 = plt.subplots() 
    
    ax2 = ax1.twinx() 

    fig.suptitle(subdir) 

    x, y = zip(*train_losses) 
    ax1.plot(x, y, label='Train Loss', c='tab:orange')
    x, y = zip(*metrics)
    ax2.plot(x, y, label=metric_name, c='tab:blue') 

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', c='tab:orange')
    ax2.set_ylabel(metric_name, c='tab:blue')


    plt.tight_layout() 

    plt.savefig('./figures/{}.png'.format(subdir))

    plt.show() 



