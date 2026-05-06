import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 位置编码类，利用正余弦函数生成位置编码矩阵，并在前向传播中将位置编码与输入的embedding相加
# 输入参数：d_model（模型维度），max_len（最大序列长度）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()
        
        # 初始化位置编码矩阵
        self.encoding=torch.zeros(max_len,d_model)
        
        # 生成位置信息（5000，1），元素为0-4999 
        position=torch.arange(0,max_len).unsqueeze(1)
        
        # 利用正余弦函数生成位置编码矩阵，元素为sin(pos/10000^(2i/d_model))和cos(pos/10000^(2i/d_model)) 
        div_term=torch.arange(0,d_model,2).float()/d_model
        self.encoding[:,0::2]=torch.sin(position/torch.pow(10000,div_term))
        self.encoding[:,1::2]=torch.cos(position/torch.pow(10000,div_term))
        # 将位置编码拓展bs->(1,5000,d_model)，方便后续与输入的embedding相加
        self.encoding=self.encoding.unsqueeze(0)
    def forward(self,x):
        # 截取出前seq_len个位置编码，并与输入的embedding相加
        x=x+self.encoding[:,:x.size(1)]
        return x
    
# 多头注意力机制类，包含Q、K、V的线性变换层和注意力机制输出后的线性变换层
# 输入参数：d_model（模型维度），n_heads（注意头的数量）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,n_heads):
        super(MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.n_heads=n_heads

        # 计算每个注意头的维度
        self.head_dim=self.d_model//self.n_heads

        # Q、K、V的Linear层，用于将输入的embedding映射到查询、键、值空间
        self.WQ=nn.Linear(d_model,d_model)
        self.WK=nn.Linear(d_model,d_model)
        self.WV=nn.Linear(d_model,d_model)

        # 注意力机制输出后的线性变换层
        self.fc=nn.Linear(d_model,d_model)

    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)

        # QKV的线性变换，将输入的embedding映射到查询、键、值空间
        Q=self.WQ(query)
        K=self.WK(key)
        V=self.WV(value)

        # 多头的拆分和转置，将Q、K、V拆分成多个头，并将维度调整为(batch_size, n_heads, seq_len, head_dim)
        Q=Q.view(batch_size,-1,self.n_heads,self.head_dim).transpose(1,2)
        K=K.view(batch_size,-1,self.n_heads,self.head_dim).transpose(1,2)
        V=V.view(batch_size,-1,self.n_heads,self.head_dim).transpose(1,2)

        # 计算注意力分数，使用缩放点积注意力机制
        socres=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
        if mask is not None:
            socres=socres.masked_fill(mask==0,float('-1e20'))
        weights=F.softmax(socres,dim=-1)

        # 将计算机权重应用到v上
        attention=torch.matmul(weights,V)
        # 将多头的输出拼接起来，并通过线性变换得到最终的输出
        attention=attention.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        output=self.fc(attention)
        return output

# 位置前馈网络类，包含两个线性变换层，中间使用ReLU激活函数
# 输入参数：d_model（模型维度），d_ff（前馈网络的维度）  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model,d_ff):
        super(PositionwiseFeedForward,self).__init__()
        # 位置前馈网络包含两个线性变换层，中间使用ReLU激活函数
        # 定义第一个线形变换层，将输入的d_model维度映射到d_ff维度
        self.fc1=nn.Linear(d_model,d_ff)
        # 定义第二个线形变换层，将d_ff维度映射回d_model维度
        self.fc2=nn.Linear(d_ff,d_model)
    def forward(self,x):
        # 前馈网络的前向传播，先通过第一个线性变换层，然后使用ReLU激活函数，最后通过第二个线性变换层
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        return x
    
# Transformer编码器层类，包含一个多头注意力机制和一个位置前馈网络，两个子层都使用残差连接和层归一化
# 输入参数：d_model（模型维度），n_heads（注意头的数量），d_ff（前馈网络的维度）
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model,n_heads,d_ff):
        super(TransformerEncoderLayer,self).__init__()
        # Transformer编码器层包含一个多头注意力机制和一个位置前馈网络，两个子层都使用残差连接和层归一化
        # 定义多头注意力机制层，输入参数为模型维度和注意头的数量
        self.multi_head_attention=MultiHeadAttention(d_model,n_heads)
        # 定义dropout层，防止过拟合，丢弃率为0.1
        self.dropout=nn.Dropout(p=0.1)
        # 定义层归一化层，输入参数为模型维度
        self.norm1=nn.LayerNorm(d_model)
        # 定义第二个层归一化层，输入参数为模型维度
        self.norm2=nn.LayerNorm(d_model)

        # 定义位置前馈网络层，输入参数为模型维度和前馈网络的维度
        self.feedforward=PositionwiseFeedForward(d_model,d_ff)
    def forward(self,x,mask=None):
        # Transformer编码器层的前向传播，首先通过多头注意力机制计算注意力输出，然后使用残差连接和层归一化，最后通过位置前馈网络计算最终输出

        # attention_out的维度为(batch_size, seq_len, d_model)，与输入x的维度相同，可以进行残差连接
        attention_out=self.multi_head_attention(x,x,x,mask)
        x=x+self.dropout(attention_out)
        x=self.norm1(x)
        x=x+self.feedforward(x)
        x=self.norm2(x)
        return x
    
#transformer Encoder Nx模块
# Transformer编码器包含多个Transformer编码器层，使用nn.ModuleList将多个编码器层组合在一起，输入参数为模型维度、注意头的数量、前馈网络的维度和编码器层的数量
# for _ in range(n_layers)表示创建n_layers个Transformer编码器层，每个编码器层的输入参数为模型维度、注意头的数量和前馈网络的维度
# Transformer编码器的前向传播，首先将输入的token序列通过嵌入层转换为embedding表示，然后添加位置编码，最后依次通过每个Transformer编码器层计算最终输出
class TransformerEncoder(nn.Module):
    def __init__(self, d_model,n_heads,n_layers,d_ff,input_vocab_size):
        super(TransformerEncoder,self).__init__()
        self.embedding=nn.Embedding(input_vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model)
        # Transformer编码器包含多个Transformer编码器层，使用nn.ModuleList将多个编码器层组合在一起，输入参数为模型维度、注意头的数量、前馈网络的维度和编码器层的数量
        # for _ in range(n_layers)表示创建n_layers个Transformer编码器层，每个编码器层的输入参数为模型维度、注意头的数量和前馈网络的维度
        self.transformer_layers=nn.ModuleList([TransformerEncoderLayer(d_model,n_heads,d_ff) for _ in range(n_layers)])
    def forward(self,x,mask=None):
        # Transformer编码器的前向传播，首先将输入的token序列通过嵌入层转换为embedding表示，然后添加位置编码，最后依次通过每个Transformer编码器层计算最终输出
        x=self.embedding(x)
        x=self.positional_encoding(x)
        for layer in self.transformer_layers:
            x=layer(x,mask)
        return x


#定义transformer解码器层类，包含一个多头注意力机制、一个位置前馈网络和一个编码器-解码器注意力机制，三个子层都使用残差连接和层归一化
#输入参数：d_model（模型维度），n_heads（注意头的数量），d_ff（前馈网络的维度）
class TransformerDecoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff):
        super(TransformerDecoderLayer,self).__init__()
        # Transformer解码器层包含一个多头注意力机制、一个位置前馈网络和一个编码器-解码器注意力机制，三个子层都使用残差连接和层归一化
        # 定义第一个多头注意力机制层，用于解码器的自注意力计算，输入参数为模型维度和注意头的数量
        self.multi_head_attention=MultiHeadAttention(d_model,n_heads)
        self.dropout=nn.Dropout(p=0.1)
        self.norm1=nn.LayerNorm(d_model)

        self.encoder_attention=MultiHeadAttention(d_model,n_heads)
        self.norm2=nn.LayerNorm(d_model)
        self.feedforward=PositionwiseFeedForward(d_model,d_ff)
        self.norm3=nn.LayerNorm(d_model)
    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
        # Transformer解码器层的前向传播，首先通过第一个多头注意力机制计算自注意力输出，然后使用残差连接和层归一化，接着通过编码器-解码器注意力机制计算与编码器输出的注意力输出，再次使用残差连接和层归一化，最后通过位置前馈网络计算最终输出
        
        multi_head_attention_out=self.multi_head_attention(x,x,x,mask=tgt_mask)
        x=x+self.dropout(multi_head_attention_out)
        x=self.norm1(x)
        encoder_attention_out=self.encoder_attention(x,encoder_output,encoder_output,mask=src_mask)
        x=x+self.dropout(encoder_attention_out)
        x=self.norm2(x)
        x=x+self.feedforward(x)
        x=self.norm3(x)
        return x
    
#transformer Decoder Nx模块
# Transformer解码器包含多个Transformer解码器层，使用nn.ModuleList将多个解码器层组合在一起，输入参数为模型维度、注意头的数量、前馈网络的维度和解码器层的数量
# for _ in range(n_layers)表示创建n_layers个Transformer解码器层，每个解码器层的输入参数为模型维度、注意头的数量和前馈网络的维度
# Transformer解码器的前向传播，首先将输入的token序列通过嵌入层转换为embedding表示，然后添加位置编码，
# 最后依次通过每个Transformer解码器层计算最终输出，并通过一个线性变换层将输出映射到词汇表大小的维度，得到最终的预测结果

class TransformerDecoder(nn.Module):
    def __init__(self, d_model,n_heads,n_layers,d_ff,output_vocab_size):    
        super(TransformerDecoder,self).__init__()
        self.embedding=nn.Embedding(output_vocab_size,d_model)
        self.positional_encoding=PositionalEncoding(d_model)
        self.transformer_layers=nn.ModuleList([TransformerDecoderLayer(d_model,n_heads,d_ff) for _ in range(n_layers)])
        self.fc_out=nn.Linear(d_model,output_vocab_size)
    def forward(self,x,encoder_output,src_mask=None,tgt_mask=None):
        x=self.embedding(x)
        x=self.positional_encoding(x)
        for layer in self.transformer_layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        output=self.fc_out(x)
        return output


# 定义Transformer模型类，包含一个编码器和一个解码器，输入参数为模型维度、注意头的数量、编码器层的数量、前馈网络的维度、输入词汇表大小和输出词汇表大小
class Transformer(nn.Module):
    def __init__(self,d_model, n_heads, n_layers, d_ff, input_vocab_size, output_vocab_size):
        super(Transformer,self).__init__()
        #编码器
        self.encoder=TransformerEncoder(d_model, n_heads, n_layers, d_ff, input_vocab_size)
        #解码器
        self.decoder=TransformerDecoder(d_model, n_heads, n_layers, d_ff, output_vocab_size)
    def forward(self,src,trg):
        #源码列mask,升维，变成(bs,1,1,seq_len)
        src_mask=src.unsqueeze(1).unsqueeze(2)
        # 通过编码器得到输出
        encoder_output = self.encoder(src, src_mask)
        # 目标序列mask
        trg_mask = self.create_target_mask(trg)
        # 通过解码器得到输出
        output = self.decoder(trg, encoder_output, src_mask, trg_mask)
        return output

    # 创建目标序列的mask， 包括填充部分和未来部分
    def create_target_mask(self,target_data):
        #形状为(bs,1,1,seq_len)
        target_pad_mask=target_data.unsqueeze(1).unsqueeze(2)
        #获取目标序列的长度，目标序列形状(bs,seq_len)
        trg_len = target_data.size(1)
        # 生成下三角矩阵，对角线及以下元素为True,其余元为False
        trg_subsequent_mask =torch.tril(torch.ones(trg_len, trg_len)).bool()
        # 将填充mask和自回归mask进行结合，逻辑与，得到最终目标序列(bs, 1, seq_len, seq_len)
        trg_mask = target_pad_mask & trg_subsequent_mask
        return trg_mask
    

if __name__ == '__main__':
# 定义一些参数或者超参数
  d_model=512# 模型的隐藏层的维度
  n_heads=8 #注意力头的数量
  n_layers=6 #transformer层的数量
  d_ff=2048#前馈神经网络中间层维度
  batch_size=32 # batch
  src_seq_length=20 #源序列长度
  trg_seq_length=10 #目标序列长度
  input_vocab_size=10000#词汇表的大小
  output_vocab_size=10000#词汇表大小
  transformer =Transformer(d_model,n_heads,n_layers,d_ff,input_vocab_size, output_vocab_size)
  # 源数据生成形状为(batch_size, src_seq_length)的随机整数，范围[0, input_vocab_size)
  src_data = torch.randint(0, input_vocab_size, (batch_size, src_seq_length))
  #目标数据
  trg_data = torch.randint(0, output_vocab_size, (batch_size, trg_seq_length))
  output = transformer(src_data, trg_data)
  print(output.shape)
  