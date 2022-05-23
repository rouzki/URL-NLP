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
    def __init__(self, model_name, num_classes, freeze_bert=False):
        super(BertCustomModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        
        # self.dropout = nn.Dropout(do_prob) ## optional
        ## 768 = self.bert.hidden_size
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
  
    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(input_ids, attention_mask)["pooler_output"] ### model outputs two hidden_state and pooled_output
        # out1 = self.dropout(pooled_output)
        output = self.classifier(pooled_output)
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
        print(self.bert.config.hidden_size)
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





def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def evaluation_scores(y_test, predicted):
    return {"Accuracy": metrics.accuracy_score(y_test, predicted, normalize=True),  # Test with normalize=False
            "Hamming loss": metrics.hamming_loss(y_test, predicted),
            "AUC": metrics.roc_auc_score(y_test, predicted),
            "F1 score macro": metrics.f1_score(y_test, predicted, average='macro', zero_division=0),
            "F1 score micro": metrics.f1_score(y_test, predicted, average='micro', zero_division=0),
            "F1 score weighted": metrics.f1_score(y_test, predicted, average='weighted', zero_division=0)}
