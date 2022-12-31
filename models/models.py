import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, input_shape, in_channel, num_hidden, filter_size):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.height, self.width = input_shape
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, self.width, self.width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, self.width, self.width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, self.width, self.width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, padding=self.padding),
            nn.LayerNorm([num_hidden, self.width, self.width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new


class ST_LSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, kernel_size, device):
        super(ST_LSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.memory = []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(SpatioTemporalLSTMCell(input_shape=self.input_shape,
                                                    in_channel=cur_input_dim,
                                                    num_hidden=self.hidden_dims[i],
                                                    filter_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j], self.memory = cell(input_, self.H[j], self.C[j], self.memory)
            else:
                self.H[j], self.C[j], self.memory = cell(self.H[j - 1], self.H[j], self.C[j], self.memory)

        return (self.H, self.C), self.H  # (hidden, output)

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
        self.memory = torch.zeros(batch_size, self.hidden_dims[0], self.input_shape[0], self.input_shape[1]).to(self.device)

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x,h_cur,c_cur):  # x [batch, hidden_dim, width, height]
     
        
   
    
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, h_next, c_next

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3),
                      stride=stride, padding=1),
            nn.GroupNorm(4, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride == 2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nin, out_channels=nout, kernel_size=(3, 3),
                               stride=stride, padding=1, output_padding=output_padding),
            nn.GroupNorm(4, nout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class image_encoder(nn.Module):
    def __init__(self, nc):
        super(image_encoder, self).__init__()
        nf = 16
        kernel_size = (3, 3)
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf * 2, stride=1)  # (nf) x 64 x 64
        self.c2 = dcgan_conv(nf * 2, nf * 4, stride=2)  # (2*nf) x 32 x 32
        self.c3 = dcgan_conv(nf * 4, nf * 8, stride=2)  # (4*nf) x 16 x 16
        self.c4 = dcgan_conv(nf * 8, nf * 16, stride=2)  # (8*nf) x 8 x 8

        self.convlstm_1 = ConvLSTM_Cell(input_dim=nf * 2, hidden_dim=nf * 2, kernel_size=kernel_size)
                  
        self.convlstm_2 = ConvLSTM_Cell(input_dim=nf * 4, hidden_dim=nf * 4, kernel_size=kernel_size)
                  
        self.convlstm_3 = ConvLSTM_Cell(input_dim=nf * 8, hidden_dim=nf * 8, kernel_size=kernel_size)
                  
        self.convlstm_4 = ConvLSTM_Cell(input_dim=nf * 16, hidden_dim=nf * 16, kernel_size=kernel_size)
                  

    def forward(self, input, H,C):
        h11 = self.c1(input)
        
      
        
        h1,H[0],C[0] = self.convlstm_1(h11, H[0],C[0])  # (nf*4) x 32 x 32
     
  
        h21 = self.c2(h1)
    
        h2,H[1],C[1] = self.convlstm_2(h21, H[1],C[1])  # (nf*4) x 32 x 32

        h31 = self.c3(h2)
    
        h3,H[2],C[2] = self.convlstm_3(h31, H[2],C[2])  # (nf*8) x 16 x 16

        h41 = self.c4(h3)
        h4,H[3],C[3] = self.convlstm_4(h41, H[3],C[3])  # (nf*16) x 8 x 8

        return H,C,[h1, h2, h3, h4]


class image_decoder(nn.Module):
    def __init__(self, nc):
        super(image_decoder, self).__init__()
        nf = 16
        kernel_size = (3, 3)
        self.upc1 = dcgan_upconv(nf * 16, nf * 8, stride=2)  # (nf*2) x 16 x 16
        self.upc2 = dcgan_upconv(nf * 8, nf * 4, stride=2)  # (nf) x 32 x 32
        self.upc3 = dcgan_upconv(nf * 4, nf * 2, stride=2)  # (nf) x 64 x 64
        self.upc4 = nn.ConvTranspose2d(nf * 2, nc, kernel_size=(3, 3), stride=1, padding=1)  # (nc) x 64 x 64

        self.convlstm_1 = ConvLSTM_Cell(input_dim=nf * 2 * 2, hidden_dim=nf * 2, kernel_size=kernel_size)
                        
        self.convlstm_2 = ConvLSTM_Cell(input_dim=nf * 4 * 2, hidden_dim=nf * 4, kernel_size=kernel_size)
                        
        self.convlstm_3 = ConvLSTM_Cell(input_dim=nf * 8 * 2, hidden_dim=nf * 8, kernel_size=kernel_size)
                        
        self.convlstm_4 = ConvLSTM_Cell(input_dim=nf * 16, hidden_dim=nf * 16, kernel_size=kernel_size)


    def forward(self, input, H,C):
        output, skip = input  # output: (4*nf) x 16 x 16
        output_4, output_3, output_2, output_1 = output
        [h1, h2, h3, h4] = skip
     
        

        d1,H[-1],C[-1] = self.convlstm_4(torch.cat([output_4], dim=1),H[-1],C[-1]) # (nf*16) x 8 x 8
        d21 = self.upc1(d1)   

       
                            # (nf*8) x 16 x 16
        d2,H[-2],C[-2] = self.convlstm_3(torch.cat([d21, output_3], dim=1),H[-2],C[-2]) # (nf*8) x 16 x 16
        d31 = self.upc2(d2)     # (nf*4) x 32 x 32
        d3,H[-3],C[-3] = self.convlstm_2(torch.cat([d31, output_2], dim=1),H[-3],C[-3]) # (nf*4) x 32 x 32
        d41 = self.upc3(d3)     # (nf*1) x 64 x 64
        d42,H[-4],C[-4] = self.convlstm_1(torch.cat([d41, output_1], dim=1),H[-4],C[-4]) # (nf*4) x 64 x 64
        d4 = self.upc4(d42)     # (nf*1) x 64 x 64

        return H,C,d4


class EncoderRNN(torch.nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        nf = 16
        kernel_size = (3, 3)
        self.device = args.device
        self.image_cnn_enc = image_encoder(16) # image encoder 64x64x1 -> 16x16x64
        self.image_cnn_dec = image_decoder(16)  # image decoder 16x16x64 -> 64x64x1

        self.convlstm_1 = ConvLSTM_Cell(input_dim=nf*2, hidden_dim=nf*2,
                                   kernel_size=kernel_size)
        self.convlstm_2 = ConvLSTM_Cell(input_dim=nf*4, hidden_dim=nf*4,
                                   kernel_size=kernel_size)
        self.convlstm_3 = ConvLSTM_Cell(input_dim=nf*8, hidden_dim=nf*8,
                                   kernel_size=kernel_size)
        self.convlstm_4 = ConvLSTM_Cell(input_dim=nf*16, hidden_dim=nf*16,
                                   kernel_size=kernel_size)
        self.nf =nf
        self.batch_size = args.batch_size


    def forward(self, input, H,C,encoder_H,encoder_C,decoder_H,decoder_C):
     
   
        encoder_H,encoder_C,skip = self.image_cnn_enc(input, encoder_H,encoder_C)
        [h1, h2, h3, h4] = skip
        output_4,H[-1],C[-1] = self.convlstm_4(h4, H[-1],C[-1])
        output_3,H[-2],C[-2] = self.convlstm_3(h3, H[-2],C[-2])
        output_2,H[-3],C[-3] = self.convlstm_2(h2, H[-3],C[-3])
        output_1,H[-4],C[-4] = self.convlstm_1(h1, H[-4],C[-4])
        output = [output_4, output_3, output_2, output_1]
        decoder_H,decoder_C,output_final = self.image_cnn_dec([output, skip], decoder_H,decoder_C)
        output_image = torch.sigmoid(output_final)

        return output_image,H,C,encoder_H,encoder_C,decoder_H,decoder_C