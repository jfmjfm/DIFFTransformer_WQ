import torch
import torch.nn as nn
import math

'''
   时间步 1: [A1, B1, C1]
   时间步 2: [A2, B2, C2]
   时间步 3: [A3, B3, C3]

位置编码会为每个时间步生成一个编码：PE1, PE2, PE3
这些编码会应用到每个时间步的所有变量上

input_embedding = self.value_embedding(x)  # x 形状: [batch, seq_len, n_variables]
position_encoding = self.position_embedding(x)  # 只依赖于 seq_len,不依赖于 n_variables
final_embedding = input_embedding + position_encoding
在多变量时间序列中，同一时间步的所有变量通常会共享相同的位置编码。这种方法使模型能够捕捉序列的时间结构，
而变量之间的关系则主要通过其他机制（如自注意力机制）来学习。   
这种位置编码应用于整个序列，不区分同一时间步的不同变量。
例如，对于序列 [[A1,B1,C1], [A2,B2,C2], [A3,B3,C3]]：
位置 1 的编码应用于 [A1,B1,C1]
位置 2 的编码应用于 [A2,B2,C2]
位置 3 的编码应用于 [A3,B3,C3]

交替使用正弦和余弦：
偶数列（0, 2, 4, ...）使用正弦函数。
奇数列（1, 3, 5, ...）使用余弦函数。
位置编码的维度：[1, max_sequence_length, d_model]
max_sequence_length 是预定义的最大序列长度
实际使用时会裁剪到当前序列的实际长度==seq_len

位置编码通常不包含批处理维度，而是在使用时广播到整个批次。
final_embedding = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
最终维度：[batch_size, sequence_length, d_model]

预测阶段，输出嵌入如何发生作用？
   def predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
       # 编码器处理输入
       enc_out = self.encoder(batch_x, batch_x_mark)
       
       # 初始化输出序列
       dec_inp = torch.zeros_like(batch_y)
       dec_inp[:, :self.label_len, :] = batch_y[:, :self.label_len, :]
       
       # 逐步预测
       for i in range(self.label_len, self.label_len + self.pred_len):
           # 输出嵌入
           dec_inp_embed = self.decoder_embedding(dec_inp, batch_y_mark)
           
           # 解码器处理
           dec_out = self.decoder(dec_inp_embed, enc_out)
           
           # 更新预测序列
           dec_inp[:, i, :] = dec_out[:, i, :]
       
       return dec_inp[:, -self.pred_len:, :]
'''

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #张量增加一个维度，变成[1,max_len,d_model]
        pe = pe.unsqueeze(0)
        #使用 self.register_buffer('pe', pe) 将位置编码注册为缓冲区。
        #缓冲区是模型状态的一部分，但不参与梯度计算。
        self.register_buffer('pe', pe)

    def forward(self, x):
        #张量变成[1,pred_len,d_model]
        #[1,96,512] #[1,144,512]
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        #in_channels是通道，卷积核仅在时间维上滑动
        #out_channels是输出通道数，等于d_model,相当于使用d_model个卷积核
        #卷积核相当于滤波器，每个滤波器提取一种特征，所以d_model个滤波器提取d_model种特征
        #一个滤波器提取一种特征，d_model个滤波器提取d_model种特征，所以d_model个滤波器提取d_model种特
        #一个滤波器得到一个标量，d_model个滤波器得到d_model个标量，所以d_model个滤波器得到d_model个标量
        #d_model个标量拼接起来，形成一个时间步的变量特征
        #seq_len个时间步的变量特征拼接起来，形成一个序列的变量特征，维度为[B,seq_len,d_model][32,96,512]

        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        #[B,L,D]->Conv[B,D,L]->[B,L,D]
        '''
        1. 输入 x 的初始形状是 [batch_size, sequence_length, c_in]
        x.permute(0, 2, 1) 将 x 转置为 [batch_size, c_in, sequence_length]，这是 Conv1d 所需的输入形状
        tokenConv 应用后，输出形状为 [batch_size, d_model, sequence_length]
        4. 最后的 transpose(1, 2) 将输出转换回 [batch_size, sequence_length, d_model]
        总结：
        tokenConv 的输入维度：[batch_size, c_in, sequence_length]
        tokenConv 的输出维度：[batch_size, d_model, sequence_length]
        transpose后[batch_size, sequence_length，d_model]
        [B,L,D]特殊情况：
        1. 卷积层（如 TokenEmbedding 中的 Conv1d）：
        输入需要调整为 [batch_size, c_in, sequence_length]
        输出后再调整回 [batch_size, sequence_length, d_model]
        某些操作可能暂时改变维度顺序，但通常会在操作后恢复到 [B, L, D] 格式。
        总结：
        在大多数 forward 函数中，维度确实保持 [B, L, D] 的格式。
        D 的具体含义（是 c_in 还是 d_model）取决于网络中的位置。
        某些特殊操作（如卷积）可能暂时改变这个顺序，但通常会迅速恢复。
        保持一致的 [B, L, D] 格式有助于模型各部分的互操作性和代码的清晰度。
        但重要的是要注意 D 的具体含义可能会随着数据在网络中的流动而变化。
            '''
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

#固定嵌入：应用于空间位置嵌入？
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        #创建一个不需要梯度的参数
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        #奇数偶数交替编码
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        #虽然 self.emb = nn.Embedding(c_in, d_model) 
        #这行代码本身不包含权重信息，但在类的初始化过程中，权重被明确地设置了
        #固定权重，且在forward中切断权重的梯度计算
        #W赋值给self.emb.weight，覆盖了原来的可训练权重
        #这种方法的优点是：
        #1. 提供了一种确定性的、基于位置的编码。
        #2. 不需要训练，减少了模型的可训练参数。
        #3. 对于某些周期性或有序的特征（如时间特征）特别有效。
        #总的来说，FixedEmbedding 类确实包含了完整的权重信息，
        #只是这些权重是预先计算好的，而不是通过学习得到的。这种
        #固定嵌入的方法在处理某些类型的输入（如时间特征）时特别有用，
        #因为它可以直接编码这些特征的内在结构，而不需要模型去学习这种结构。
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        #固定编码不计算梯度，detach切断梯度计算
        #detach() 创建一个新的张量，与原始张量共享数据但不共享计算历史。
        #新张量不会跟踪其计算历史，因此不会参与反向传播。

        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

'''
为什么需要 y_mark 的时间嵌入:
预测未来值时，模型需要知道它正在为哪些时间点进行预测。
时间嵌入提供了重要的时间上下文信息，如周期性、季节性等。
在预测阶段,y_mark 通常包含两部分：
a. 已知的标签部分（对应 label_len)
b. 未来的预测部分（对应 pred_len)
c. batch_y_mark 需要包含整个输出序列的时间特征，包括已知的标签部分和未来的预测部分
'''
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
#传统Transformer嵌入,值嵌入采用卷积嵌入，时间序列位置嵌入采用正弦嵌入，时间嵌入采用时间特征嵌入
#最终采用加法合成一个嵌入
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):  
        #print('x,y',x.shape)#[1,96,21][1,144,21] in test and pred, [32,96,21] in train
        #print('x_mark,y_mark',x_mark.shape)#[1,96,4][1,144,4] in test and pred, [32,96,4] in train
        #print('self.value_embedding(x),self.value_embedding(y)',self.value_embedding(x).shape)#[1,96,512]
        #print('self.position_embedding(x),self.position_embedding(y)',self.position_embedding(x).shape)#[1,96,512]
        #print('self.temporal_embedding(x_mark),self.temporal_embedding(y_mark)',self.temporal_embedding(x_mark).shape)#[1,96,512]  
        #print('self.value_embedding(x) + self.position_embedding(x),self.value_embedding(y) + self.position_embedding(y)',(self.value_embedding(x) + self.position_embedding(x)).shape)#[1,96,512]
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)#[1,96,512] in test and pred, [32,96,512] in train
        return self.dropout(x)


#inverse 专用，使用了全连接嵌入
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        #没有使用卷积原因，它善于捕捉局部特征，但可能不如全连接层擅长建模长期依赖关系。
        #全连接层可以处理更复杂的模式，尤其是在数据具有非线性关系时。
        #值嵌入采用全连接嵌入,在时间序列维度上进行嵌入，投影到d_model维度
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        # 如果有时间特征，则合并。问题是，什么时候没有时间特征？SOLAR数据集？
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            #沿着变量维度拼接，然后全连接嵌入，投影到d_model维度
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        # 时间特征维度为4，变量特征维度21，变量特征在前，时间特征在后。
        #print('DataEmbedding_inverted',x.shape)#[32,21+5,512][1,（21+4）,512]
        return self.dropout(x)

