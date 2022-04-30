import torch
import torchvision.models as models
from torch import nn


def create_resnet_basic_block(
    width_output_feature_map, height_output_feature_map, nb_channel_in, nb_channel_out
):
    basic_block = nn.Sequential(
        nn.Upsample(size=(width_output_feature_map, height_output_feature_map), mode="nearest"),
        nn.Conv2d(
            nb_channel_in,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            nb_channel_out,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
    )
    return basic_block


class Model_Segmentation_Traffic_Light_Supervised(nn.Module):
    def __init__(
        self,
        nb_images_input,
        nb_images_output,
        hidden_size,
        nb_class_segmentation,
        nb_class_dist_to_tl,
        crop_sky=False,
    ):
        super().__init__()
        if crop_sky:
            self.size_state_RL = 6144
        else:
            self.size_state_RL = 8192
        resnet18 = models.resnet18(pretrained=False)

        # See https://arxiv.org/abs/1606.02147v1 section 4: Information-preserving
        # dimensionality changes
        #
        # "When downsampling, the first 1x1 projection of the convolutional branch is performed
        # with a stride of 2 in both dimensions, which effectively discards 75% of the input.
        # Increasing the filter size to 2x2 allows to take the full input into consideration,
        # and thus improves the information flow and accuracy."

        assert resnet18.layer2[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer3[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer4[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        resnet18.layer2[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer3[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer4[0].downsample[0].kernel_size = (2, 2)

        assert resnet18.layer2[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer3[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer4[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        new_conv1 = nn.Conv2d(
            nb_images_input * 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet18.conv1 = new_conv1

        self.encoder = torch.nn.Sequential(
            *(list(resnet18.children())[:-2])
        )  # resnet18_no_fc_no_avgpool
        self.last_conv_downsample = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.fc1_traffic_light_inters = nn.Linear(self.size_state_RL, hidden_size)
        self.fc2_tl_inters_none = nn.Linear(
            hidden_size, 4
        )  # Classification traffic_light US, traffic_light EU, intersection, none
        self.fc2_traffic_light_state = nn.Linear(
            hidden_size, 2
        )  # Classification red/orange or green (maybe 3 classes in stead of 2?)
        self.fc2_distance_to_tl = nn.Linear(
            hidden_size, nb_class_dist_to_tl
        )  # classification on the distance to traffic_light

        self.fc1_delta_y_yaw_camera = nn.Linear(
            self.size_state_RL, int(hidden_size / 4)
        )  # Hard coded here, we want a little hidden size...
        self.fc2_delta_y_yaw_camera = nn.Linear(
            int(hidden_size / 4), 2 * nb_images_output
        )  # Regression on delta_y and delta_yaw of each input frame!

        # We will upsample image with nearest neightboord interpolation between each umsample block
        # https://distill.pub/2016/deconv-checkerboard/
        self.up_sampled_block_0 = create_resnet_basic_block(6, 8, 512, 512)
        self.up_sampled_block_1 = create_resnet_basic_block(12, 16, 512, 256)
        self.up_sampled_block_2 = create_resnet_basic_block(24, 32, 256, 128)
        self.up_sampled_block_3 = create_resnet_basic_block(48, 64, 128, 64)
        self.up_sampled_block_4 = create_resnet_basic_block(74, 128, 64, 32)

        self.last_conv_segmentation = nn.Conv2d(
            32,
            nb_class_segmentation * nb_images_output,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
        )
        self.last_bn = nn.BatchNorm2d(
            nb_class_segmentation * nb_images_output,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, batch_image):
        # Encoder first, resnet18 without last fc and abg pooling
        encoding = self.encoder(batch_image)  # 512*4*4 or 512*4*3 (crop sky)

        encoding = self.last_conv_downsample(encoding)

        # Segmentation branch
        upsample0 = self.up_sampled_block_0(encoding)  # 512*8*8 or 512*6*8 (crop sky)
        upsample1 = self.up_sampled_block_1(upsample0)  # 256*16*16 or 256*12*16 (crop sky)
        upsample2 = self.up_sampled_block_2(upsample1)  # 128*32*32 or 128*24*32 (crop sky)
        upsample3 = self.up_sampled_block_3(upsample2)  # 64*64*64 or 64*48*64 (crop sky)
        upsample4 = self.up_sampled_block_4(upsample3)  # 32*128*128 or 32*74*128 (crop sky)

        out_seg = self.last_bn(
            self.last_conv_segmentation(upsample4)
        )  # nb_class_segmentation*128*128

        # Classification branch, traffic_light (+ state), intersection or none
        classif_state_net = encoding.view(-1, self.size_state_RL)

        traffic_light_state_net = self.fc1_traffic_light_inters(classif_state_net)
        traffic_light_state_net = nn.functional.relu(traffic_light_state_net)

        classif_output = self.fc2_tl_inters_none(traffic_light_state_net)
        state_output = self.fc2_traffic_light_state(traffic_light_state_net)
        dist_to_tl_output = self.fc2_distance_to_tl(traffic_light_state_net)

        delta_position_yaw_state = self.fc1_delta_y_yaw_camera(classif_state_net)
        delta_position_yaw_state = nn.functional.relu(delta_position_yaw_state)
        delta_position_yaw_output = self.fc2_delta_y_yaw_camera(delta_position_yaw_state)

        return out_seg, classif_output, state_output, dist_to_tl_output, delta_position_yaw_output
