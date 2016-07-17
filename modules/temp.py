'''
        this function sets the encoder states, given the source_seq_numpy as vector (:,)
        '''
        xt_source = numpy.dot(
            self.model['Emb_source'][source_seq_numpy, :],
            self.model['Emb_tune_source']
        )
        shape_encode = xt_source.shape
        ht_source = numpy.zeros(
            shape_encode, dtype = dtype
        )
        ct_source = numpy.zeros(
            shape_encode, dtype = dtype
        )
        # assume ht is same size with xt --
        # both projected to same space
        len_source, dim_model = shape_encode[0], shape_encode[1]
        for time_stamp in range(-1, len_source-1, 1):
            post_transform = numpy.dot(
                numpy.concatenate(
                    (
                        xt_source[time_stamp+1, :], ht_source[time_stamp, :]
                    ), axis=0
                ),
                self.model['W_recur_source']
            )
            #
            gate_input_numpy = self.sigmoid(
                post_transform[:self.dim_model]
            )
            gate_forget_numpy = self.sigmoid(
                post_transform[self.dim_model:2*self.dim_model]
            )
            gate_output_numpy = self.sigmoid(
                post_transform[2*self.dim_model:3*self.dim_model]
            )
            gate_pre_c_numpy = numpy.tanh(
                post_transform[3*self.dim_model:]
            )
            ct_source[time_stamp+1, :] = gate_forget_numpy * ct_source[time_stamp, :] + gate_input_numpy * gate_pre_c_numpy
            #ht_source[time_stamp+1, :] = gate_output_numpy * ct_source[time_stamp+1, :]
            ht_source[time_stamp+1, :] = gate_output_numpy * numpy.tanh(ct_source[time_stamp+1, :])
            #
        self.ht_encode = ht_source[-1, :]
        self.ct_encode = ct_source[-1, :]



def _validatestep(actid, postfeat):
        if actid == 3:
            return True
        elif actid == 1:
            return True
        elif actid == 2:
            return True
        elif actid == 0:
            if postfeat[23] > 0.5:
                return False
            else:
                return True
        else:
            print "impossible action dude!"





def init_beam(self, pos_start, pos_end):
    print "initialize beam ... "
    item  = {
        'htm1': numpy.copy(self.ht_encode),
        'ctm1': numpy.copy(self.ct_encode),
        'list_idx_action': [],
        #
        'pos_current': pos_start,
        'pos_destination': pos_end,
        'list_pos': [numpy.copy(pos_start)]
        #
        'feat_current_position': numpy.copy(
            self.seq_world_numpy[0, :]
        ),
        #
        'continue': True,
        #
        'cost': 0.00
    }
    self.beam_list.append(item)
