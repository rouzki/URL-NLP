import pandas as pd
import numpy as np
import pickle5
import random, os
from tqdm.notebook import tqdm, trange
import torch
import transformers
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics

from utils import URLDataset, BertCustomModel, loss_fn, evaluation_scores

###### SET VARIABLES
DATA_PATH = "../data/"
RANDOM_SEED = 12345
MAX_LEN = 40
bs = 32
lr = 2e-5
PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-uncased'
# PRE_TRAINED_MODEL_NAME = "camembert-base"
# PRE_TRAINED_MODEL_NAME = "flaubert/flaubert_base_cased"
EPOCHS = 5
OUT_DIR = 'models_trained/bertm/'
MODEL_SAVING_NAME = OUT_DIR  + 'model_finetuned.h5'

### usign GPU if avilllabel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"USING DEVICE {device}")

print("Reading Data")
###### READING DATA 
with open(DATA_PATH  + 'data_cleaned.pickle', 'rb') as handle:
    data = pickle.load(handle)


### set seed
def seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed(RANDOM_SEED)

### MultiLabel Binzarizer
mlb = MultiLabelBinarizer()
y_transformed = mlb.fit_transform(data['target_cleaned'].values)
class_names = mlb.classes_

###### Splitting DATA
print("Splitting Data")
X, X_test, y, y_test = train_test_split(data['url_cleaned'], y_transformed, test_size=0.1, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

### Get Tokenizer
print("Getting Bert Model/Tokenizer")
print(f"Choosed model is {PRE_TRAINED_MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

### Create model
model = BertCustomModel(model_name = PRE_TRAINED_MODEL_NAME, num_classes = len(class_names))
model.to(device);

### tokenize
print("### Tokenizing text data (urls)")
train_encodings = tokenizer(list(X_train.values), truncation=True, add_special_tokens=True, max_length=MAX_LEN, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt')
val_encodings = tokenizer(list(X_val.values), truncation=True, add_special_tokens=True, max_length=MAX_LEN, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt')
test_encodings = tokenizer(list(X_test.values), truncation=True, add_special_tokens=True, max_length=MAX_LEN, return_token_type_ids=False, padding='max_length', return_attention_mask=True, return_tensors='pt')

### Convert text to URL DATASET
train_dataset = URLDataset(train_encodings, y_train)
val_dataset = URLDataset(val_encodings, y_val)
test_dataset = URLDataset(test_encodings, y_test)


### Create data loaders
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)




### Optimizer
optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr)

print("Model started training")
#### START TRAINING MODEL
training_scores = []
best_auc = 0
## for each epoch
for epoch in trange(EPOCHS, desc="Epoch"):

    train_loss = 0
    valid_loss = 0

    model.train()
    print('#### Training epoch {}   ####'.format(epoch))
 
    ## for each bach 
    for batch in tqdm(train_loader, leave=False, desc="Batch"):

        ## getting model inputs
        input_ids = batch['input_ids'].to(device, dtype = torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.float)

        outputs = model(input_ids, attention_mask=attention_mask)

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
        train_loss += loss.item()

    train_loss=train_loss/len(train_loader)

    print('#### Validating epoch {}   ####'.format(epoch))

    eval_loss = 0.0
    model.eval()

    eval_labels = []
    eval_outputs = []

    with torch.no_grad():
      for batch_idx, batch_val in enumerate(val_loader, 0):

            ids = batch_val['input_ids'].to(device, dtype = torch.long)
            mask = batch_val['attention_mask'].to(device, dtype = torch.long)
            labels = batch_val['labels'].to(device, dtype = torch.float)

            outputs = model(ids, mask)

            loss = loss_fn(outputs, labels)
            eval_loss += loss.item()

            eval_labels.extend(labels.cpu().detach().numpy().tolist())
            eval_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())


    eval_outputs = np.array(eval_outputs) >= 0.5
    scores = evaluation_scores(eval_labels, eval_outputs)
    
    # print(f"AUC Score = {auc}")
    if scores['AUC'] > best_auc:
        torch.save(model, MODEL_SAVING_NAME)
        # model.save_pretrained(OUT_DIR)
        # tokenizer.save_pretrained(MODEL_PATH)
        best_auc = scores['AUC']


    avg_train_loss, avg_val_loss = train_loss / len(train_loader), eval_loss / len(val_loader)

    ### Scoring
    eval_scores =  {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
        }
    eval_scores.update(scores)
    training_scores.append(eval_scores)
    print(eval_scores)



print("##### Start doing model evaluation ie:e evaluation is done on test set ##############")
## set model on evaluation mode
model.eval()



agg_prediction = []
agg_labels = []

with torch.no_grad():
    for batch_idx, batch_test in enumerate(test_loader, 0):

        
        input_ids = batch_test['input_ids'].to(device, dtype = torch.long)
        mask = batch_test['attention_mask'].to(device, dtype = torch.long)
        labels = batch_test['labels'].to(device, dtype = torch.float)

        outputs = model(input_ids, mask)

        pred_probs = torch.sigmoid(outputs).detach().cpu().numpy()
        
        pred_bools = np.where(pred_probs > 0.5, 1, 0)
        
        agg_prediction.append(pred_bools)
        agg_labels.append(labels.to('cpu').numpy())
    
    pred_labels = [item for element in agg_prediction for item in element]
    true_labels = [item for element in agg_labels for item in element]


test_scores = evaluation_scores(true_labels, pred_labels)
print(test_scores)
print("saving data for inference")


with open(OUT_DIR + 'mlb.pickle', 'wb') as handle:
    pickle5.dump(mlb, handle, protocol=pickle5.HIGHEST_PROTOCOL)

with open(OUT_DIR + 'model_scoring.pickle', 'wb') as handle:
    pickle5.dump([training_scores, test_scores], handle, protocol=pickle5.HIGHEST_PROTOCOL)
