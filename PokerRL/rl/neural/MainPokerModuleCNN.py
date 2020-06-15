# 2020 Vsevolod Kompantsev

import torch
import torch.nn as nn


class MainPokerModuleCNN(nn.Module):
    """
    Uses structure of FLAT module but substitutes FC layers with CNN+Dropout layers
    Uses Leaky ReLU instead of ReLU
    Works only with bucketed preflop hands
    """

    def __init__(self,
                 env_bldr,
                 device,
                 mpm_args,
                 ):
        super().__init__()

        self.args = mpm_args

        self.env_bldr = env_bldr

        self.N_SEATS = self.env_bldr.N_SEATS
        self.device = device

        self.board_start = self.env_bldr.obs_board_idxs[0]
        self.board_stop = self.board_start + len(self.env_bldr.obs_board_idxs)

        self.pub_obs_size = self.env_bldr.pub_obs_size
        self.priv_obs_size = self.env_bldr.priv_obs_size

        self._relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        if mpm_args.use_pre_layers:
            self.cards_cn_1 = nn.Conv2d(in_channels=1,
                                        out_channels=mpm_args.card_block_units, kernel_size=(2, 2),
                                        padding=1)
            self.cards_cn_2 = nn.Conv2d(in_channels=mpm_args.card_block_units,
                                        out_channels=mpm_args.card_block_units, kernel_size=(2, 2),
                                        padding=0)
            self.cards_cn_3 = nn.Conv2d(in_channels=mpm_args.card_block_units,
                                        out_channels=mpm_args.card_block_units, kernel_size=(2, 2),
                                        padding=1)
            self.cards_cn_4 = nn.Conv2d(in_channels=mpm_args.card_block_units,
                                        out_channels=mpm_args.card_block_units, kernel_size=(2, 2),
                                        padding=0)
            self.cards_cn_5 = nn.Conv2d(in_channels=mpm_args.card_block_units,
                                        out_channels=1, kernel_size=(1, 1), stride=1,
                                        padding=0)
            #self.mpool = nn.MaxPool2d((2, 2), padding=0)

            self.final_fc_1 = nn.Linear(in_features=192, out_features=mpm_args.other_units)

        else:
            raise ValueError('CNN requires pre_layers to be enabled')

        self.lut_range_idx_2_priv_o = torch.from_numpy(self.env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS)
        self.lut_range_idx_2_priv_o = self.lut_range_idx_2_priv_o.to(device=self.device, dtype=torch.float32)

        self.lut_range_idx_2_priv_o_pf = torch.from_numpy(self.env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS_PF)
        self.lut_range_idx_2_priv_o_pf = self.lut_range_idx_2_priv_o_pf.to(device=self.device, dtype=torch.float32)

        self.to(device)

    @property
    def output_units(self):
        return self.args.other_units

    def forward(self, pub_obses, range_idxs):
        """
        1. bucket hands if round is preflop
        2. feed through pre-processing fc layers
        3. PackedSequence (sort, pack)
        4. fully connected nn
        5. unpack (unpack re-sort)

        Args:
            pub_obses (float32Tensor):                 Tensor of shape [torch.tensor([history_len, n_features]), ...)
            range_idxs (LongTensor):        range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
        """

        # ____________________________________________ Packed Sequence _____________________________________________
        # we bucket preflop hands, discarding suits info
        # if game state = preflop, which is stored in pub_obses[:,14].
        # To do so we use another pre-created idx-obses table, where all suits are 0

        priv_obses = self.lut_range_idx_2_priv_o[range_idxs]
        pf_mask = torch.where(pub_obses[:, 14] == 1)
        priv_obses[pf_mask] = self.lut_range_idx_2_priv_o_pf[range_idxs][pf_mask]

        if self.args.use_pre_layers:
            priv_obses = torch.reshape(priv_obses, (-1, 2, 17))
            _board_obs = pub_obses[:, self.board_start:self.board_stop]
            _board_obs = torch.reshape(_board_obs, (-1, 5, 17))
            _card_obs = torch.cat((priv_obses, _board_obs), dim=1)
            _hist_and_state_obs = torch.cat([
                pub_obses[:, :self.board_start],
                pub_obses[:, self.board_stop:]
            ],
                dim=-1
            )
            _hist_and_state_obs.unsqueeze_(1)
            _card_obs = nn.ZeroPad2d((0, 7, 0, 0))(_card_obs)
            _card_obs = torch.cat((_card_obs, _hist_and_state_obs), dim=1)
            # Add dimension for convolution channels
            _card_obs.unsqueeze_(1)
            y = self._feed_through_pre_layers(card_obs=_card_obs,
                                              hist_and_state_obs=_hist_and_state_obs)

        else:
            raise ValueError('CNN requires pre_layers to be enabled')

        final = y.flatten(1)
        final = self._relu(self.final_fc_1(final))

        # Standartize last layer
        if self.args.normalize:
            means = final.mean(dim=1, keepdim=True)
            stds = final.std(dim=1, keepdim=True)
            final = (final - means) / stds

        return final

    def _feed_through_pre_layers(self, card_obs, hist_and_state_obs):

        # """""""""""""""
        # Cards Body
        # """""""""""""""
        cards_out1 = self._relu(self.cards_cn_1(card_obs))
        cards_out2 = self._relu(self.cards_cn_2(cards_out1))
        cards_out3 = self._relu(self.cards_cn_3(cards_out2) + cards_out1)
        cards_out4 = self._relu(self.cards_cn_4(cards_out3) + cards_out2)
        cards_out = self._relu(self.cards_cn_5(cards_out4))

        return cards_out


class MPMArgsCNN:

    def __init__(self,
                 use_pre_layers=True,
                 card_block_units=192,
                 other_units=64,
                 normalize=True,
                 dropout=0.25
                 ):
        self.use_pre_layers = use_pre_layers
        self.other_units = other_units
        self.card_block_units = card_block_units // 4
        self.normalize = normalize
        self.dropout = dropout

    def get_mpm_cls(self):
        return MainPokerModuleCNN
