import torch
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, channels=512, label_size=31):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        in_channels, out_channels = embed_size, channels
        self.layers = nn.ModuleList()

        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                        )
                    )
        
        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                        )
        )

        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                        )
                    )
        
        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1*2, dilation=2, bias=False),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                        )
        )

        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1*2, dilation=2, bias=False),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                        )
        )

        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=3*2, dilation=2, bias=False),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            nn.Dropout(p=0.4),
                        )
        )

        self.layers.append(
                        nn.Sequential(
                            nn.Conv1d(out_channels, 16, kernel_size=5, stride=1, padding=3*2, dilation=2, bias=False),
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.Dropout(p=0.4),
                        )
        )

        self.avg_pool  = nn.Sequential(nn.AdaptiveAvgPool1d(4))
        self.last = nn.Sequential(  nn.Dropout(p=0.5),
                                    nn.Flatten(), ) 

        self.linear = nn.Linear(in_features = 64, out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """
        #print(x.shape)
        #print(x)
        input = self.embed(x)
        input = input.transpose(1,2)    # (B,T,H) -> (B,H,T)
        
        #CNN
        for i in self.layers:
            # input = i(input)
            input = self.maskconv1d(i(input),lengths)
        #Adapative
        dummy = [] 
        for k,i in enumerate(input):
            dummy.append(self.avg_pool(i[:,:lengths[k]].view(1,-1,lengths[k])))
        input = self.last(torch.cat(dummy,0))
        del dummy

        #print(cnn.shape)
        logits = self.linear(input)

        return logits
        
    def maskconv1d(self,x,lengths):
        # print(x.shape)
        max_len = x.size(2)
        idxs = torch.arange(max_len).to(lengths.dtype).to(lengths.device).expand(len(lengths), max_len)
        mask = idxs >= lengths.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        # print(mask.shape)
        # print(x.shape)
        
        del mask, idxs, max_len 
        return x


class ICASSP3CNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=512, num_lstm_layers = 2, bidirectional = False, label_size=31):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional
        
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(embed_size, embed_size, kernel_size=7, padding=3)

        self.batchnorm = nn.BatchNorm1d(3 * embed_size)

        self.lstm = nn.LSTM(input_size = 3 * embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """

        batch_size = x.shape[0]
        input = self.embed(x)
        
        batch_size = input.size(0)
        input = input.transpose(1,2)    # (B,T,H) -> (B,H,T)

        cnn_output = torch.cat([self.cnn(input), self.cnn2(input), self.cnn3(input)], dim=1)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits
        
        
class ICASSP2CNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=512, num_lstm_layers = 2, bidirectional = False, label_size=31):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional
        
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)

        self.batchnorm = nn.BatchNorm1d(2 * embed_size)

        self.lstm = nn.LSTM(input_size = 2 * embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """

        input = self.embed(x)
        
        batch_size = input.size(0)
        input = input.transpose(1,2)    # (B,T,H) -> (B,H,T)

        cnn_output = torch.cat([self.cnn(input), self.cnn2(input)], dim=1)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits
    

class ICASSP1CNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=512, num_lstm_layers = 2, bidirectional = False, label_size=31):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional
        
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)

        self.batchnorm = nn.BatchNorm1d(embed_size)

        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """

        input = self.embed(x)
        
        batch_size = input.size(0)
        input = input.transpose(1,2)    # (B,T,H) -> (B,H,T)

        cnn_output = self.cnn(input)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]
        
        logits = self.linear(h_n)

        return logits
        
