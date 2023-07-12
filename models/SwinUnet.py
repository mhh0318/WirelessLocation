import torch
import torch.nn as nn
import pytorch_lightning as pl


class TokenToCoordinate(nn.Module):
    def __init__(self, input_channels, L):
        super(TokenToCoordinate, self).__init__()
        self.conv = nn.Conv1d(input_channels, 1, kernel_size=1)
        self.linear = nn.Linear(L, 2)

    def forward(self, x):
        b= x.shape[0]
        x = x.permute(0, 2, 1)  # B, C, L
        x = self.conv(x)  # B, 2, L
        x = x.view(b,-1)  # B, L, 2
        x = self.linear(x)  # B, 2
        return x

class Swin2D(pl.LightningModule):

    def __init__(self, swin, learning_rate=1e-4):#2.7542287033381663e-05
        super().__init__()
        self.model = swin
        # self.input_conv = torch.nn.Conv2d(16,3,1,1)        # self.mass_prediction = torch.nn.Conv1d(64, 2, kernel_size=1)
        # self.pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        self.loss = torch.nn.MSELoss()
        self.act = torch.nn.GELU()
        self.learning_rate = learning_rate

        self.predict = TokenToCoordinate(48, 64**2)


    def get_input(self, batch):
        feat = batch['feat'] * 2. - 1. # 0 - 1
        # feat = batch['feat']
        # gt = batch['gt'] / 127.5 - 1. # 0 - 255
        gt = batch['gt'] / 255.
        # gt = batch['gt'] 
        return feat, gt

    def forward(self, feat):
        h = self.model(feat)
        output = self.predict(h)
        output = self.act(output)
        return output

    def training_step(self, train_batch, batch_idx):
        x, y = self.get_input(train_batch)
        coor = self.forward(x)
        if y.max() > 1.:
            coor_real = coor
            y_real = y
        else:
            coor_real = coor*255.
            y_real = y*255.
        loss = self.loss(coor,y)
        # loss = self.loss(coor_real,y_real)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if y.max() <= 1.:
            self.log('real_loss',self.loss(coor_real,y_real))

        if self.global_step % 50 == 0:


            outputs_sample = coor_real[:8].detach().cpu().numpy()
            labels_sample = y_real[:8].detach().cpu().numpy()


            table_text_outputs = '<table><tr><th>Sample</th>' + ''.join(f'<th>{i}</th>' for i in range(len(outputs_sample))) + '</tr><tr><th>x_pred</th>' + ''.join(f'<td>{x:.2f}</td>' for x, y in outputs_sample) + '</tr><tr><th>y_pred</th>' + ''.join(f'<td>{y:.2f}</td>' for x, y in outputs_sample) + '</tr></table>'
            table_text_labels = '<table><tr><th>Sample</th>' + ''.join(f'<th>{i}</th>' for i in range(len(labels_sample))) + '</tr><tr><th>x_gt</th>' + ''.join(f'<td>{x:.0f}</td>' for x, y in labels_sample) + '</tr><tr><th>y_gt</th>' + ''.join(f'<td>{y:.0f}</td>' for x, y in labels_sample) + '</tr></table>'


            self.logger.experiment.add_text('outputs', table_text_outputs, self.global_step)
            self.logger.experiment.add_text('labels', table_text_labels, self.global_step)


        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,20,45,70], gamma=0.1)
        return optimizer
        # return ([optimizer],[lr_scheduler])
    
    def validation_step(self, val_batch, batch_idx):
        x, y = self.get_input(val_batch)
        coor = self.forward(x)
        if y.max() > 1.:
            coor_real = coor
            y_real = y
        else:
            coor_real = coor*255.
            y_real = y*255.
        loss = self.loss(coor_real, y_real)
        self.log('val_loss', loss)
        return loss
