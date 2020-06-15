# 2020 Vsevolod Kompantsev

import torch
import torch.nn as nn


class MainPokerModuleFLAT2(nn.Module):
    """
    Feeds parts of the observation through different fc layers
    Main idea is to make the network deeper and maintain the same train ability and computation cost.
    First is achieved with skip connections, the second - with 25% lower layer width, see division in MPMArgsFLAT2

    Structure (each branch merge is a concat):

    Table & Player state --> FC -> RE -> FCS -> RE ------------------------------------------------------.
    Board Cards ---> FC -> RE --> cat -> FC -> RE -> FCS -> RE -> FCS -> RE -> FCS -> RE -> FC -> RE --> cat --> FC -> RE -> FC -> RE -> FCS-> RE -> FC -> RE -> FCS-> RE ->Standartize
    Private Cards -> FC -> RE -'


    where FCS refers to FC+Skip and RE refers to leaky ReLU
    Note that the final layer is skip-connected with the stride of 1, FC1 to FC3, FC3 to FC5
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

        self._relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)  # was ReLU

        if mpm_args.use_pre_layers:

            self._priv_cards = nn.Linear(in_features=self.env_bldr.priv_obs_size,
                                         out_features=mpm_args.other_units)
            self._board_cards = nn.Linear(in_features=self.env_bldr.obs_size_board,
                                          out_features=mpm_args.other_units)
            self._cards_bn_small = nn.Identity() #nn.BatchNorm1d(mpm_args.other_units)

            self.cards_fc_1 = nn.Linear(in_features=2 * mpm_args.other_units, out_features=mpm_args.card_block_units)
            self.cards_fc_2 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.card_block_units)
            self.cards_fc_3 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.card_block_units)
            self.cards_fc_4 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.card_block_units)
            self.cards_fc_5 = nn.Linear(in_features=mpm_args.card_block_units, out_features=mpm_args.other_units)
            self._cards_bn_big = nn.Identity() #nn.BatchNorm1d(mpm_args.card_block_units)

            self.hist_and_state_1 = nn.Linear(in_features=self.env_bldr.pub_obs_size - self.env_bldr.obs_size_board,
                                              out_features=mpm_args.other_units)
            self.hist_and_state_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

            self.final_fc_1 = nn.Linear(in_features=2 * mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_3 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_4 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_5 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

        else:
            self.final_fc_1 = nn.Linear(in_features=2 * self.env_bldr.complete_obs_size, out_features=mpm_args.other_units)
            self.final_fc_2 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_3 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_4 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)
            self.final_fc_5 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

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
            _board_obs = pub_obses[:, self.board_start:self.board_stop]
            _hist_and_state_obs = torch.cat([
                pub_obses[:, :self.board_start],
                pub_obses[:, self.board_stop:]
            ],
                dim=-1
            )
            y = self._feed_through_pre_layers(board_obs=_board_obs, priv_obs=priv_obses,
                                              hist_and_state_obs=_hist_and_state_obs)

        else:
            y = torch.cat((priv_obses, pub_obses,), dim=-1)

        final = self._relu(self.final_fc_1(y))
        final2 = self._relu(self.final_fc_2(final))
        final3 = self._relu(self.final_fc_3(final2) + final)
        final4 = self._relu(self.final_fc_4(final3))
        final = self._relu(self.final_fc_5(final4) + final3)

        # Standartize last layer
        if self.args.normalize:
            means = final.mean(dim=1, keepdim=True)
            stds = final.std(dim=1, keepdim=True)
            final = (final - means) / stds

        return final

    def _feed_through_pre_layers(self, priv_obs, board_obs, hist_and_state_obs):

        # """""""""""""""
        # Cards Body
        # """""""""""""""
        _priv_1 = self._relu(self._priv_cards(priv_obs))
        _board_1 = self._relu(self._board_cards(board_obs))

        cards_out = self._relu(self.cards_fc_1(torch.cat([_priv_1, _board_1], dim=-1)))
        cards_out2 = self._relu(self.cards_fc_2(cards_out))
        cards_out3 = self._relu(self.cards_fc_3(cards_out2) + cards_out)
        cards_out4 = self._relu(self.cards_fc_4(cards_out3) + cards_out3)
        cards_out = self.cards_fc_5(cards_out4)

        hist_and_state_out = self._relu(self.hist_and_state_1(hist_and_state_obs))
        hist_and_state_out = self.hist_and_state_2(hist_and_state_out) + hist_and_state_out

        return self._relu(torch.cat([cards_out, hist_and_state_out], dim=-1))


class MPMArgsFLAT2:

    def __init__(self,
                 use_pre_layers=True,
                 card_block_units=192,
                 other_units=64,
                 normalize=True,
                 dropout=0.25
                 ):
        self.use_pre_layers = use_pre_layers
        self.other_units = other_units // 4 * 3
        self.card_block_units = card_block_units // 4 * 3
        self.normalize = normalize
        self.dropout = dropout

    @staticmethod
    def get_mpm_cls():
        return MainPokerModuleFLAT2
