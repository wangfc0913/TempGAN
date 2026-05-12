import torch
from torch import nn
from .base_model import BaseModel
from . import networks
from TeVnet.models import TeVNet


class TempGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # =========================
        # Loss & Visuals
        # =========================
        self.loss_names = ['G_GAN', 'D', 'G_L1']
        self.visual_names = ['real_A', 'real_B', 'fake_B']

        # =========================
        # Models
        # =========================
        self.model_names = ['G', 'D']

        # Generator
        # 使用原 BicycleGAN Generator
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.nz,
            opt.ngf,
            netG=opt.netG,
            norm=opt.norm,
            nl=opt.nl,
            use_dropout=opt.use_dropout,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            gpu_ids=self.gpu_ids,
            where_add=opt.where_add,
            upsample=opt.upsample
        )

        # Discriminator
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        self.netD = networks.define_D(
            D_output_nc,
            opt.ndf,
            netD=opt.netD,
            norm=opt.norm,
            nl=opt.nl,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            num_Ds=opt.num_Ds,
            gpu_ids=self.gpu_ids
        )

        # =========================
        # Loss functions
        # =========================
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # Optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.tev_loss = TeVNet()

    # =========================
    # Data I/O
    # =========================
    def set_input(self, input):
        """
        A: ε(x,y)  pixel-wise emissivity
        B: I_T     infrared image at temperature T
        T: scalar environment temperature
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_T = input['T'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_name = input['image_name']

    # =========================
    # Forward
    # =========================
    def get_temperature_embedding(self, T):
        """
        T: (B,) or (B,1)
        return: (B, nz)
        """
        T = T.float()
        if T.dim() == 1:
            T = T.unsqueeze(1)
        T = 2 * (T + 9) / 39 - 1
        z_T = T.repeat(1, self.opt.nz)
        return z_T

    def temperature_sensitivity_loss(self, gen_rad, temp):
        """
        重构温度敏感度损失：直接约束同一batch内，温度高的样本辐射值更高
        无需依赖普朗克定律，避免维度/尺度问题
        """
        temp_sorted, idx_sorted = torch.sort(temp)
        gen_rad_sorted = gen_rad[idx_sorted]
        rad_diff = gen_rad_sorted[1:] - gen_rad_sorted[:-1]
        loss = torch.mean(torch.clamp(-rad_diff, min=0))
        return loss

    def test(self, T0=None):
        with torch.no_grad():
            if T0 is not None:
                if not isinstance(T0, torch.Tensor):
                    T0 = torch.tensor(T0, device=self.device, dtype=torch.float32)
                    if T0.dim() == 0:  # 标量 -> [1, 1]
                        T0 = T0.unsqueeze(0).unsqueeze(0)
                    elif T0.dim() == 1:  # 1维 -> [batch_size, 1]
                        T0 = T0.unsqueeze(1)
                self.real_T = T0.to(self.device)

            z_T = self.get_temperature_embedding(self.real_T)
            self.fake_B = self.netG(self.real_A, z_T)
            return self.real_A, self.fake_B, self.real_B

    def test_pair(self):
        z_T = self.get_temperature_embedding(self.real_T)
        self.fake_B = self.netG(self.real_A, z_T)
        return self.real_A, self.fake_B, self.real_B, self.image_name

    def forward(self):
        """
        Conditional forward:
        G(ε, T) → I_T
        """
        # ---- Generator ----
        z_T = self.get_temperature_embedding(self.real_T)  # (B, nz)

        self.fake_B = self.netG(self.real_A, z_T)

        # ---- Discriminator input ----
        if self.opt.conditional_D:
            self.fake_data = torch.cat([self.real_A, self.fake_B], dim=1)
            self.real_data = torch.cat([self.real_A, self.real_B], dim=1)
        else:
            self.fake_data = self.fake_B
            self.real_data = self.real_B

    # =========================
    # Discriminator
    # =========================
    def backward_D(self):
        pred_fake = self.netD(self.fake_data.detach())
        pred_real = self.netD(self.real_data)

        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)

        self.loss_D = loss_D_fake + loss_D_real
        self.loss_D.backward()

    # =========================
    # Generator
    # =========================
    def backward_G(self):
        # GAN loss
        pred_fake = self.netD(self.fake_data)
        self.loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN *= self.opt.lambda_GAN

        # L1 reconstruction loss (physical consistency)
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(
                self.fake_B, self.real_B
            ) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.sen_loss = self.temperature_sensitivity_loss(self.fake_B, self.real_T)
        self.loss_tev = self.tev_loss(self.real_A, self.fake_B, self.real_T)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + 1 * self.sen_loss + 1 * self.loss_tev
        self.loss_G.backward()

    # =========================
    # Optimization
    # =========================
    def update_G(self):
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_D(self):
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()
