

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
            self._priv_cards = nn.Conv1d(in_channels=1,
                                         out_channels=mpm_args.card_block_units, kernel_size=3, stride=1, padding=1)
            self._board_cards = nn.Conv1d(in_channels=1,
                                          out_channels=mpm_args.card_block_units, kernel_size=3, stride=1,
                                          padding=1)

            self.cards_cn_1 = nn.Conv1d(in_channels=2 * mpm_args.card_block_units,
                                        out_channels=2 * mpm_args.card_block_units, kernel_size=3, stride=1,
                                        padding=1)
            self.cards_cn_2 = nn.Conv1d(in_channels=2 * mpm_args.card_block_units,
                                        out_channels=mpm_args.card_block_units, kernel_size=3, stride=1,
                                        padding=1)
            self.cards_cn_3 = nn.Conv1d(in_channels=mpm_args.card_block_units,
                                        out_channels=mpm_args.other_units, kernel_size=3, stride=1,
                                        padding=1)

            self.hist_and_state_1 = nn.Conv1d(in_channels=1,
                                              out_channels=mpm_args.other_units, kernel_size=4, stride=1, padding=1)
            self.hist_and_state_2 = nn.Conv1d(in_channels=mpm_args.other_units,
                                              out_channels=mpm_args.other_units, kernel_size=4, stride=1, padding=1)

            self.final_cn_1 = nn.Conv1d(in_channels=mpm_args.other_units,
                                        out_channels=mpm_args.other_units, kernel_size=4, stride=1, padding=1)
            self.final_fc_1 = nn.Linear(in_features=560, out_features=mpm_args.other_units)

        else:
            self.final_cn_1 = nn.Conv1d(in_channels=1,
                                        out_channels=mpm_args.other_units, kernel_size=4, stride=1, padding=1)
            self.final_fc_1 = nn.Linear(in_features=mpm_args.other_units, out_features=mpm_args.other_units)

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

            # Add dimension for conv1d's channels
            _board_obs.unsqueeze_(1)
            priv_obses.unsqueeze_(1)
            _hist_and_state_obs.unsqueeze_(1)
            y = self._feed_through_pre_layers(board_obs=_board_obs, priv_obs=priv_obses,
                                              hist_and_state_obs=_hist_and_state_obs)

        else:
            y = torch.cat((priv_obses, pub_obses,), dim=-1)

        final = self._relu(self.final_cn_1(y))
        final = final.flatten(1)
        final = self._relu(self.final_fc_1(final))

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

        cards_out = self._relu(self.cards_cn_1(torch.cat([_priv_1, _board_1], dim=-1)))
        cards_out = self._relu(self.cards_cn_2(cards_out))
        cards_out = self.cards_cn_3(cards_out)

        hist_and_state_out = self._relu(self.hist_and_state_1(hist_and_state_obs))
        hist_and_state_out = self.hist_and_state_2(hist_and_state_out)

        return self._relu(torch.cat([cards_out, hist_and_state_out], dim=-1))


class MPMArgsCNN:

    def __init__(self,
                 use_pre_layers=True,
                 card_block_units=192,
                 other_units=64,
                 normalize=True,
                 dropout=0.25
                 ):
        self.use_pre_layers = use_pre_layers
        self.other_units = other_units // 16
        self.card_block_units = card_block_units // 16
        self.normalize = normalize
        self.dropout = dropout

    def get_mpm_cls(self):
        return MainPokerModuleCNN
