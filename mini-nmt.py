from _gdynet import *
# from _dynet import *
import numpy as np

class Translator(object):

    def __init__(self, **kwargs):
        self.layers = kwargs.pop('lstm_num_of_layers', 2)
        self.semb_dim = kwargs.pop('semb_dim1', 32)
        self.temb_dim = kwargs.pop('temb_dim1', 32)
        self.hid_dim = kwargs.pop('hid_dim1', 32)
        self.att_dim = kwargs.pop('att_dim1', 32)
        self.svoc_size = kwargs.pop('src_vocab_size', 1000)
        self.tvoc_size = kwargs.pop('trg_vocab_size', 1000)
        self.drop_rate = kwargs['dropout']

        self.src_nhids = self.hid_dim
        self.trg_nhids = self.hid_dim

        print 'build lstm model and init parameters ...',
        self.build_lstm_model()
        print 'done'

    def build_lstm_model(self):
        self.model = Model()
        self.trainer = AdadeltaTrainer(self.model, eps=1e-7, rho=0.99)

        self.enc_fwd_lstm = LSTMBuilder(self.layers, self.semb_dim, self.src_nhids, self.model)
        self.enc_bck_lstm = LSTMBuilder(self.layers, self.semb_dim, self.src_nhids, self.model)

        self.dec_lstm = LSTMBuilder(self.layers, self.src_nhids, self.trg_nhids, self.model)

        # memory...
        self.s_lookup_table = self.model.add_lookup_parameters((self.svoc_size, self.semb_dim))
        self.s_lookup_table.init_from_array(
            np.random.uniform(-0.01, 0.01, (self.svoc_size, self.semb_dim)))

        self.t_lookup_table = self.model.add_lookup_parameters((self.tvoc_size, self.temb_dim))
        self.t_lookup_table.init_from_array(
            np.random.uniform(-0.01, 0.01, (self.tvoc_size, self.temb_dim)))

        # r = np.sqrt(6. / (shape[0] + shape[1]))
        unfiniter = UniformInitializer(0.01)
        self.attention_w1 = self.model.add_parameters((self.att_dim, 2 * self.src_nhids), unfiniter)
        self.attention_w2 = self.model.add_parameters(
            (self.att_dim, 2 * self.layers * self.trg_nhids), unfiniter)
        self.attention_v = self.model.add_parameters((1, self.att_dim), ConstInitializer(0.))

        self.att_t_w = self.model.add_parameters((self.trg_nhids, 2 * self.src_nhids), unfiniter)
        self.att_t_b = self.model.add_parameters((self.trg_nhids), ConstInitializer(0.))

        self.dec_w_init = self.model.add_parameters((self.trg_nhids, 2 * self.src_nhids), unfiniter)
        self.dec_b_init = self.model.add_parameters((self.trg_nhids), ConstInitializer(0.))

        # self.no_voc_w = self.model.add_parameters(
        #    (self.svoc_size, self.semb_dim + 2 * self.src_nhids + self.trg_nhids), unfiniter)
        self.no_voc_w = self.model.add_parameters(
            (self.svoc_size, self.temb_dim + self.trg_nhids + self.trg_nhids), unfiniter)
        self.no_voc_b = self.model.add_parameters((self.svoc_size), ConstInitializer(0.))
    def s_sent_id2emb(self, sentence):
        return [self.s_lookup_table[wid] for wid in sentence]

    def t_sent_id2emb(self, sentence):
        return [self.t_lookup_table[wid] for wid in sentence]

    def run_lstm(self, initial_state, input_vecs):
        s = initial_state
        outs = []
        for vec in input_vecs:
            s = s.add_input(vec)
            output = s.output()
            outs.append(output)
        return outs

    def bi_enc_sentence(self, sentence):
        inv_sentence = list(reversed(sentence))  # len*(emb, batch)
        fwd_enc_vec = self.run_lstm(
            self.enc_fwd_lstm.initial_state(), sentence)
        bck_enc_vec = self.run_lstm(
            self.enc_bck_lstm.initial_state(), inv_sentence)
        bck_enc_vec = list(reversed(bck_enc_vec))
        # src_len * (2*src_nhids,)
        return [concatenate(list(p)) for p in zip(fwd_enc_vec, bck_enc_vec)]

    def attend(self, ctx_vecs, state):
        w1 = parameter(self.attention_w1)
        w2 = parameter(self.attention_w2)
        v = parameter(self.attention_v)
        concat_state_output = concatenate(list(state.s()))  # s:(trg_nhids,)
        logger.debug('concatenate of {} {}'.format(type(concat_state_output),
                                                   concat_state_output.npvalue().shape))
        # concatenate the current state and current output
        w2a1 = w2 * concat_state_output     # (att_dim,)
        weights = []
        for ctx_vec in ctx_vecs:
            # ctx_vec: (2*src_nhids,)
            weight_one_sword = v * tanh(w1 * ctx_vec + w2a1)   # (1,) * batch
            weights.append(weight_one_sword)
        weights_probs = softmax(concatenate(weights))   # src_len
        ctx_with_attend = esum(
            [c * p for c, p in zip(ctx_vecs, weights_probs)])
        # print 'context with attention', ctx_with_attend.npvalue().shape
        # (2*src_nhids,) * batch   weighted average over source sentence length
        # weighted average over all encoded vector (2*src_hids,) of source words
        return ctx_with_attend

    def forward(self, ctx_vecs, ref):
        w = parameter(self.no_voc_w)
        b = parameter(self.no_voc_b)

        s0 = self.dec_lstm.initial_state()
        s = s0.add_input(vecInput(self.hid_dim * 2))  # (2*hid_dim,)
        losses = []
        for wid in ref:
            ctx_with_attend = self.attend(ctx_vecs, s)  # (2*hid_dim,)

            s = s.add_input(ctx_with_attend)
            # print s.output().npvalue().shape
            # print type(w)
            # print type(s.output())
            out = w * s.output() + b    # (svoc_size, 1)
            dist_voc = softmax(out)
            losses.append(-log(pick(dist_voc, wid)))
        sumloss = esum(losses)
        return sumloss

    import configurations

    def translate(self, src_sent, ref_sent, eos_idx=0, svoc=None, tvoc=None):
        def sample(probs):
            rnd = np.random.random()
            for i, p in enumerate(probs):
                rnd -= p
                if rnd <= 0:
                    break
            return i

        s_embedded = self.s_sent_id2emb(src_sent)
        print len(s_embedded), s_embedded[0].npvalue().shape
        encoded = self.bi_enc_sentence(s_embedded)
        print len(encoded), encoded[0].npvalue().shape

        t_embedded = self.t_sent_id2emb(ref_sent)
        print len(t_embedded), t_embedded[0].npvalue().shape

        w = parameter(self.no_voc_w)
        b = parameter(self.no_voc_b)
        a2tw = parameter(self.att_t_w)
        a2tb = parameter(self.att_t_b)

        prevy_embbed = inputVector(np.zeros((self.temb_dim,)))

        s0 = self.dec_lstm.initial_state()
        start_input = self.init_dec_input(encoded)

        print start_input.npvalue().shape
        s = self.dec_lstm.initial_state().add_input(start_input)
        out_idvec = []
        maxlen = len(src_sent) * 2
        for i in range(maxlen):
            ctx_with_attend = self.attend(encoded, s)
            print ctx_with_attend.npvalue().shape
            ctx_with_attend = a2tw * ctx_with_attend + a2tb
            print ctx_with_attend.npvalue().shape
            s = s.add_input(ctx_with_attend)
            concat = concatenate([prevy_embbed, ctx_with_attend, s.output()])
            print concat.npvalue().shape
            out = w * concat + b  # (voc, ) * batch
            print out.npvalue().shape
            probs = softmax(out)
            probs = probs.vec_value()
            nwid = sample(probs)    # randomly
            prevy_embbed = self.t_lookup_table[nwid]
            if nwid == eos_idx:
                break
            out_idvec.append(nwid)
        print '[src] {}'.format(' '.join([svoc[i] for i in src_sent]))
        print '[ref] {}'.format(' '.join([tvoc[i] for i in ref_sent]))
        print '[out] {}'.format(' '.join([tvoc[i] for i in out_idvec]))
        return out_idvec

    def sent_loss(self, x, y):
        # renew_cg()
        emb_x = self.s_sent_id2emb(x)
        enc = self.bi_enc_sentence(emb_x)    # (srclen, 2*hid_dim)
        sentloss = self.forward(enc, y)
        return sentloss

    def init_dec_input(self, xbienc_lhb, bxm=None):
        w = parameter(self.dec_w_init)
        b = parameter(self.dec_b_init)
        np_ctx = []
        for c in xbienc_lhb:
            np_ctx.append(c.npvalue())
        np_ctx = np.array(np_ctx)
        if bxm is not None:
            bxm = np.transpose(bxm)   # (len, batch)
            ctx_mean = (np_ctx * bxm[:, None, :]).sum(0) / bxm.sum(0)
        else:
            ctx_mean = np_ctx.mean(0)
        # F means to flatten in column-major order, as dynet, (2*src_nhids, batch)
        ctx_dy = inputMatrix(ctx_mean.flatten('F').tolist(), ctx_mean.shape)
        # init_input = tanh(w * ctx_dy + b)    # why doesn't this work?
        if ctx_dy.npvalue().ndim == 1:
            init_input = tanh(w * ctx_dy + b)
        else:
            init_input = tanh(colwise_add(w * ctx_dy, b))
        logger.debug('decoder initial input {} {}'.format(
            type(init_input), init_input.npvalue().shape))
        return init_input

    def convert_dec_state(self, ctx_watt, s, bxm=None):
        w_cz = parameter(self.dec_w_cz)
        w_hz = parameter(self.dec_w_hz)
        b_z = parameter(self.dec_b_z)

        w_cr = parameter(self.dec_w_cr)
        w_hr = parameter(self.dec_w_hr)
        b_r = parameter(self.dec_b_r)

        w_ch = parameter(self.dec_w_ch)
        w_hh = parameter(self.dec_w_hh)
        b_h = parameter(self.dec_b_h)

        s = s.s()[0]
        # colwise_add(w_cz * ctx_watt, b_z)
        logger.debug('w_cz: {} {}'.format(type(w_cz), w_cz.npvalue().shape))
        logger.debug('attention {} {}'.format(
            type(ctx_watt), ctx_watt.npvalue().shape))
        # z = logistic(w_cz * ctx_watt + w_hz * s + b_z)    # sigmoid
        # r = logistic(w_cr * ctx_watt + w_hr * s + b_r)    # sigmoid

        # zw = w_cz * ctx_watt + w_hz * s
        # ctx_watt_dy = matInput(cshape[0], cshape[1])
        # ctx_watt_set(ctx_watt_npvalue().flatten())    # problem here...
        # by column set
        ctx_watt_dy = exp_to_input(ctx_watt)
        s_dy = exp_to_input(s)
        z = logistic(colwise_add(w_cz * ctx_watt_dy + w_hz * s_dy, b_z))
        # z = logistic(colwise_add((w_cz * ctx_watt + w_hz * s), b_z))
        # # sigmoid
        r = logistic(colwise_add(
            w_cr * ctx_watt_dy + w_hr * s_dy, b_r))    # sigmoid
        c_h = w_ch * ctx_watt_dy

        hidden = tanh(cwise_multiply(
            colwise_add(w_hh * s_dy, b_h), r) + c_h)
        print 'z:', type(z)
        print z.npvalue().shape
        hidden = cwise_multiply(s_dy, z) + cwise_multiply((1. - z), hidden)

        bxm = np.transpose(bxm)
        hidden_np = hidden.npvalue()
        if bxm is not None:
            hidden = bxm[:, None] * hidden_np + (1. - bxm)[:, None] * s.npvalue()
        hshape = hidden.shape     # (src_len, batch)
        hidden_dy = vecInput(hshape[0] * hshape[1])
        hidden_dy.set(hidden.flatten())
        hidden_dy = transpose(reshape(hidden_dy, (hshape[1], hshape[0])))
        return hidden_dy

    def upd_loss_wbatch_with_atten_sbs(self, bx, bxm, by, bym):
        # (src_len, batch_size)
        renew_cg()
        w = parameter(self.no_voc_w)
        b = parameter(self.no_voc_b)
        a2tw = parameter(self.att_t_w)
        a2tb = parameter(self.att_t_b)

        minibatch_sizex, src_len = bx.shape
        minibatch_sizey, trg_len = by.shape
        assert(minibatch_sizex == minibatch_sizey)
        minibatch_size = minibatch_sizex
        # Batch lookup (for every timestamp of RNN or LSTM)
        logger.debug('start one batch ...')
        emb_batch_x = [lookup_batch(
            self.s_lookup_table, [bx[i][j] for i in range(minibatch_size)]) for j in range(src_len)]
        # src_len * (semb_dim, batch_size), type(_gdynet._lookupBatchExpression)
        logger.debug('lookuptable => source length {}'.format(len(emb_batch_x)))
        logger.debug('embedding {} {}'.format(
            type(emb_batch_x[0]), emb_batch_x[0].npvalue().shape))

        emb_batch_y = [lookup_batch(
            self.t_lookup_table, [by[i][j] for i in range(minibatch_size)]) for j in range(trg_len)]
        # trg_len * (temb_dim, batch_size), type(_gdynet._lookupBatchExpression)

        # src_len*(2*hid_dim, 80)
        xbienc_lhb = self.bi_enc_sentence(emb_batch_x)
        logger.debug('bi-encode => source length {}'.format(len(xbienc_lhb)))
        logger.debug(
            'bi-encode => {} {}'.format(type(xbienc_lhb[0]), xbienc_lhb[0].npvalue().shape))

        s0 = self.dec_lstm.initial_state()

        # randomly initializing starting input
        # s = s0.add_input(matInput(self.hid_dim * 2, minibatch_size))
        # s = s0.add_input(vecInput(self.hid_dim))

        start_input = self.init_dec_input(xbienc_lhb, bxm=None)
        start_input = reshape(start_input, (self.trg_nhids,), batch_size=minibatch_size)
        # start_input = reshape(start_input, self.hid_dim, batch_size=minibatch_size)
        logger.debug('after convert batch, start input {} {}'.format(
            type(start_input), start_input.npvalue().shape))
        # print start_input.npvalue()

        s = s0.add_input(start_input)
        total_reflen_batch = 0
        logger.debug('decode state {} {}'.format(
            type(s.h()[-1]), s.h()[-1].npvalue().shape))

        # because we need to use attention, so we can not transduce here
        # lstm_outputs = self.dec_lstm.initial_state().transduce(xbienc_lhb)
        # print 'lstm output length in this batch', len(lstm_outputs)
        # print 'lstm output batch shape', lstm_outputs[0].npvalue().shape
        lstm_outputs, losses = [], []
        for wid in range(trg_len):
            # if by == 0:
            #    continue

            emb_y = reshape(
                emb_batch_y[wid], (self.temb_dim,), batch_size=minibatch_size)

            y_filter = filter(lambda x: x != 0, by[wid])
            total_reflen_batch += len(y_filter)

            ctx_with_attend = self.attend(xbienc_lhb, s)
            # (2*src_nhids, ) * batch_size tensor
            ctx_with_attend = a2tw * ctx_with_attend + a2tb
            # (trg_nhids, ) * batch_size tensor
            ctx_with_attend = reshape(
                ctx_with_attend, (self.trg_nhids,), batch_size=minibatch_size)
            logger.debug('context attention batch {} {}'.format(type(ctx_with_attend),
                                                                ctx_with_attend.npvalue().shape))
            # s = self.convert_dec_state(ctx_with_attend, s, bxm)
            # s.set_h([ctx_with_attend])    # s.output() is (32,80), because s.s() are all (32,80)
            # print s.h()[0].npvalue().shape
            # print s.output().npvalue().shape
            # out = self.convert_dec_state(ctx_with_attend, s, bxm)
            # s.set_h([out])
            step_lstm_out = s.output()  # (trg_nhids,) * b
            logger.debug('each step lstm output => {} {}'.format(
                type(step_lstm_out), step_lstm_out.npvalue().shape))
            lstm_outputs.append(step_lstm_out)

            s = s.add_input(ctx_with_attend)
            # (temb_dim+src_nhids+trg_nhids, ) * batch
            concat = concatenate([emb_y, ctx_with_attend, step_lstm_out])
            out = w * concat + b  # (voc, ) * batch
            out = transpose(reshape(out, (self.svoc_size,), batch_size=minibatch_size))  # (voc,1,b)
            out = transpose(out)
            logger.debug('final output: {}'.format(out.npvalue().shape))
            logger.debug(list(by[:, wid].flatten()))
            bloss = pickneglogsoftmax_batch(out, list(by[:, wid].flatten()))
            sum_loss = sum_batches(bloss)
            losses.append(sum_loss)
        for l in losses:
            logger.debug('one column words in each batch loss => {} {} {}'.format(
                type(l), l.npvalue().shape, l.npvalue()))
        # print 'sum => ', type(esum(losses)), esum(losses).npvalue().shape, esum(losses).npvalue()
        # zw = average(losses)
        # print type(losses)
        # print 'avg => ', type(average(losses)), average(losses).npvalue().shape
        avg = esum(losses) / len(losses)
        return avg, minibatch_size, total_reflen_batch
