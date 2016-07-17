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

def _getleftright(direc):
        left = direc - 90
        if left == -90:
            left = 270
        right = direc + 90
        if right == 360:
            right = 0
        behind = direc + 180
        if behind == 360:
            behind = 0
        elif behind == 450:
            behind = 90
        return left, right, behind

 def _onestep(post, thisdirec):
        # just one step forward, does not mean turn
        nextpos = np.copy(post)
        if thisdirec  == 0:
            nextpos[1] -= 1
        elif thisdirec  == 90:
            nextpos[0] += 1
        elif thisdirec  == 180:
            nextpos[1] += 1
        elif thisdirec  == 270:
            nextpos[0] -= 1
        else:
            print "no valid direction in beam search?"
        return nextpos

    def _step(post, actid, thismap):
        #
        nextpos = np.zeros((3,),dtype=np.int)
        if actid == 1: # turn left -- always possible
            nextpos[0] = post[0]
            nextpos[1] = post[1]
            turnleft, turnright, _ = _getleftright(post[2])
            nextpos[2] = turnleft
        elif actid == 2:
            nextpos[0] = post[0]
            nextpos[1] = post[1]
            turnleft, turnright, _ = _getleftright(post[2])
            nextpos[2] = turnright
        elif actid == 3:
            nextpos[0] = post[0]
            nextpos[1] = post[1]
            nextpos[2] = post[2]
        elif actid == 0:
            # move forward, no need to check whether it can go
            # since already validated before come to here
            nextpos = _onestep(post, post[2])
        else:
            print "no such action? wrong!"
        return nextpos



def _getfeat(startpos, thismap):
        startx = startpos[0]
        starty = startpos[1]
        startdirec = startpos[2]
        # we can definitely find this position since validation is already done
        # when we get this position
        thisfeat = np.zeros((78,),dtype=dtype)
        somelab = 0
        for j, node in enumerate(maps[thismap]['nodes']):
            if ( node['x']==startx and node['y']==starty ):
                thisleft, thisright, thisbehind = _getleftright(startdirec)
                #
                nodefeat = np.copy(np.cast[dtype](node['objvec']) )
                forwardfeat = np.copy(node['capfeat'][startdirec])
                leftfeat = np.copy(node['capfeat'][thisleft])
                rightfeat = np.copy(node['capfeat'][thisright])
                behindfeat = np.copy(node['capfeat'][thisbehind])
                #
                thisfeat = np.concatenate((nodefeat,forwardfeat,leftfeat,rightfeat, behindfeat),axis=0)
                #
                #thisfeat = \
                #np.concatenate((node['objvec'],node['nbfeats'][startdirec],node['nbfeats'][thisleft],node['nbfeats'][thisright]),
                #               axis=0)
                somelab += 2
        local = np.copy(np.cast[dtype](thisfeat) )
        if somelab > 1:
            somelab += 0 # do thing, since we find it
        else:
            # it is wrong, since we did not find this position in this map
                # did not find it, that is wrong
            print "no such a postion in this map? -- while beam search !!!"
        return local




def init_beam(self, pos_start, pos_end):
    print "initialize beam ... "
    item  = {
        'htm1': numpy.copy(self.ht_encode),
        'ctm1': numpy.copy(self.ct_encode),
        'list_idx_action': [],
        'list_pos': [numpy.copy(pos_start)],
        #
        'pos_current': pos_start,
        'pos_destination': pos_end,
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
