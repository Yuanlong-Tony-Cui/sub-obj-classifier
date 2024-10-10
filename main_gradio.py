
import torch
import torchtext
import gradio as gr
import torch.nn as nn



'''
    This BaselineModel is from <Baseline.ipynb>.
'''
class BaselineModel(nn.Module):
    def __init__(self, vocab):
        super(BaselineModel, self).__init__()

        # Embedding layer:
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=True)

        # Fully-connected layer:
        embedding_dim = vocab.vectors.shape[1]
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Note: x is an array of word indices, which is batch_size by max_seq_length

        # Step 1: Translate word indices to word embeddings:
        # print('x.shape:', x.shape)
        embeddings = self.embedding(x)

        # Step 2: Compute the average embedding:
        # print('embeddings.shape', embeddings.shape)
        avg_embedding = embeddings.mean(dim=1)  # batch_size by embedding_dim
        # print('avg_embedding.shape', avg_embedding.shape)

        # Step 3: Pass it through the fully-connected layer:
        output = self.fc(avg_embedding)

        # Note: We do NOT apply sigmoid here since BCEWithLogitsLoss includes it already

        return output
    


'''
    This CNNClassifier is from <CNN.ipynb>.
    NOTE: We have to change the default values of the parameters
    to match what we saved in the checkpoint file.
'''
class CNNClassifier(nn.Module):
  def __init__(self, vocab, k1=4, k2=4, n1=20, n2=20):
    super(CNNClassifier, self).__init__()

    self.embedding = nn.Embedding.from_pretrained(vocab.vectors, freeze=True)
    
    # Convolutional layers:
    embedding_dim = vocab.vectors.shape[1]
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=n1, kernel_size=(k1, embedding_dim), bias=False)
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=n2, kernel_size=(k2, embedding_dim), bias=False)
    
    # Fully-connected layer:
    # Input size: n1 + n2
    self.fc = nn.Linear(n1 + n2, 1)

  def forward(self, x):
    # Expand word indices to word embeddings:
    embeddings = self.embedding(x)
    embeddings = embeddings.unsqueeze(1)

    # Use ReLU:
    conv1_out = nn.functional.relu(self.conv1(embeddings))
    conv2_out = nn.functional.relu(self.conv2(embeddings))

    pool1_out = nn.functional.max_pool2d(conv1_out, kernel_size=(conv1_out.size(2), 1)).squeeze(2)
    pool2_out = nn.functional.max_pool2d(conv2_out, kernel_size=(conv2_out.size(2), 1)).squeeze(2)

    combined = torch.cat((pool1_out, pool2_out), dim=1)
    combined = combined.squeeze(2)

    output = self.fc(combined)

    return output
  


'''
    Helper function (used from the NLP course)
'''
def split_input_string(sentence):

    tokens = sentence.split()

    # Convert to integer representation per token
    token_ints = [glove.stoi.get(tok, len(glove.stoi)-1) for tok in tokens]

    # Convert into a tensor of the shape accepted by the models
    token_tensor = torch.LongTensor(token_ints).view(-1,1)

    return token_tensor



'''
    The `fn` to be used in `gr.Interface()`
'''
def get_classification_results(sentence):
    # Load the baseline model using checkpoint:
    baseline_model = BaselineModel(vocab=glove)
    checkpoint = torch.load('baseline.pt', weights_only=True)
    baseline_model.load_state_dict(checkpoint)
    baseline_model.eval()

    # Load the baseline model using checkpoint:
    cnn_model = CNNClassifier(vocab=glove)
    checkpoint_cnn = torch.load('cnn.pt', weights_only=False)
    cnn_model.load_state_dict(checkpoint_cnn)
    cnn_model.eval()
 
    # Use `squeeze()` and `unsqueeze()` to reshape the input tensor:
    input_tensor = split_input_string(sentence).squeeze().unsqueeze(0)
    # print('input_tensor.shape', input_tensor.shape)

    prob_baseline = torch.sigmoid(baseline_model.forward(input_tensor)).item()
    prob_cnn = torch.sigmoid(cnn_model.forward(input_tensor)).item()

    # Determine the class:
    class_baseline = "Subjective"
    if prob_baseline > 0.5:
        class_baseline = "Subjective"
    else:
        class_baseline = "Objective"
    
    class_cnn = "Subjective"
    if prob_cnn > 0.5:
        class_cnn = "Subjective"
    else:
        class_cnn = "Objective"

    # Format the probability:
    prob_baseline_formatted = f"{prob_baseline:.4f}"
    prob_cnn_formatted = f"{prob_cnn:.4f}"
    
    return class_baseline, prob_baseline_formatted, class_cnn, prob_cnn_formatted



'''
    Build and launch Gradio:
'''

glove = torchtext.vocab.GloVe(name="6B",dim=100) # embedding size = 100

demo = gr.Interface(
    fn=get_classification_results,
    inputs=gr.Textbox(label="Input Sentence"),
    outputs=[
        gr.Textbox(label="Classification given by the baseline model"),
        gr.Textbox(label="Probability given by the baseline model"),
        gr.Textbox(label="Classification given by the CNN model"),
        gr.Textbox(label="Probability given by the CNN model"),
    ],
    title="Subjective/Objective Sentence Classification",
    description="Enter a sentence to classify it as subjective or objective with the output probabilities."
)

demo.launch()

