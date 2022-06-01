import torch
from transformers import AutoModel
from sklearn import metrics

class URLDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



class BertCustomModel(torch.nn.Module):
    """
    CUSTOM FUNCTION to use Bert with a classifier
    classifier is a linear layer with dropout 
    */ params /*
    - concat_hidden_states:
    if we want to apply a concatenation on the all 4 layers hidden states
    - freeze_bert:
    function to Freeze bert (bert params not updates during training) ie: train only the classifier.
    """

    def __init__(self, model_name, num_classes, freeze_bert=False, concat_hidden_states = False):
        super(BertCustomModel, self).__init__()
        self.concat_hidden_states = concat_hidden_states

        ### bert + option to return all hidden states in case of concatenation
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=self.concat_hidden_states)

        self.dropout = torch.nn.Dropout(0.1) ## optional
        

        if self.concat_hidden_states: ## multiply number of hidden size by 4 only if we concatenated all the 4 layers hidden states
          self.classifier = torch.nn.Linear(self.bert.config.hidden_size * 4, num_classes)
        else: self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

        
        ## if we chose to freeze bert
        if freeze_bert:
            # self.bert.weight.requires_grad_(False) ### onother option to do it
            for param in self.bert.parameters():
                param.requires_grad = False
  
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask) ### model output: hidden_state [1, nbr_tokens, 768] + pooled_output [1, 1, 768]

        #### concat hidden states
        if self.concat_hidden_states:
            ## concatenating the hidden states of the last four layers, taking the output from [CLS], 
            hidden_states = bert_output['hidden_states']
            pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
            pooled_output = pooled_output[:, 0, :]
        else:
            ## get only pooler output
            pooled_output = bert_output["pooler_output"]
        
        out = self.dropout(pooled_output)
        output = self.classifier(out)
        return output



################# Special class for FlauBert ########################################
from transformers import FlaubertConfig
from transformers.modeling_utils import SequenceSummary

class FlauBertCustomModel(torch.nn.Module):
    def __init__(self, model_name, num_classes, freeze_bert=False):
        super(FlauBertCustomModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        ### using sequence summary cause Flaubert is XLM bert type
        self.sequence_summary = SequenceSummary(FlaubertConfig)
        
        # self.dropout = nn.Dropout(do_prob) ## optional
        # print(self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
  
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask) ### model outputs two hidden_state and pooled_output
        # out1 = self.dropout(pooled_output)
        logits = self.sequence_summary(pooled_output[0])
        output = self.classifier(logits)
        return output
#######################################################################################



### loss function

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

### calculate necessairy fucntions
def evaluation_scores(y_test, predicted):
    return {"Accuracy": metrics.accuracy_score(y_test, predicted, normalize=True),  # Test with normalize=False
            "Hamming loss": metrics.hamming_loss(y_test, predicted),
            "AUC": metrics.roc_auc_score(y_test, predicted),
            "F1 score macro": metrics.f1_score(y_test, predicted, average='macro', zero_division=0),
            "F1 score micro": metrics.f1_score(y_test, predicted, average='micro', zero_division=0),
            "F1 score weighted": metrics.f1_score(y_test, predicted, average='weighted', zero_division=0)}