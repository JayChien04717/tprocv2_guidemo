
def acquire(self, soc, soft_avgs=1, load_pulses=True, start_src="internal", threshold=None, angle=None, progress=True, st_progress=None, remove_offset=True):
     """Acquire data using the accumulated readout.

        Parameters
        ----------
        soc : QickSoc
            Qick object
        soft_avgs : int
            number of times to rerun the program, averaging results in software (aka "rounds")
        load_pulses : bool
            if True, load pulse envelopes
        start_src: str
            "internal" (tProc starts immediately) or "external" (each round waits for an external trigger)
        threshold : float or list of float
            The threshold(s) to apply to the I values after rotation.
            Length-normalized units (same units as the output of acquire()).
            If scalar, the same threshold will be applied to all readout channels.
            A list must have length equal to the number of declared readout channels.
        angle : float or list of float
            The angle to rotate the I/Q values by before applying the threshold.
            Units of radians.
            If scalar, the same angle will be applied to all readout channels.
            A list must have length equal to the number of declared readout channels.
        progress: bool
            if true, displays progress bar
        remove_offset: bool
            Some readouts (muxed and tProc-configured) introduce a small fixed offset to the I and Q values of every decimated sample.
            This subtracts that offset, if any, before returning the averaged IQ values or rotating to apply software thresholding.

        Returns
        -------
        numpy.ndarray
            averaged IQ values (float)
            divided by the length of the RO window, and averaged over reps and rounds
            if threshold is defined, the I values will be the fraction of points over threshold
            dimensions for a simple averaging program: (n_ch, n_reads, 2)
            dimensions for a program with multiple expts/steps: (n_ch, n_reads, n_expts, 2)
        """
      # don't load memories now, we'll do that later
      self.config_all(soc, load_pulses=load_pulses, load_mem=False)

       if any([x is None for x in [self.counter_addr, self.loop_dims, self.avg_level]]):
            raise RuntimeError(
                "data dimensions need to be defined with setup_acquire() before calling acquire()")

        # configure tproc for internal/external start
        soc.start_src(start_src)

        n_ro = len(self.ro_chs)

        total_count = functools.reduce(operator.mul, self.loop_dims)
        self.acc_buf = [np.zeros((*self.loop_dims, nreads, 2), dtype=np.int64)
                        for nreads in self.reads_per_shot]
        self.stats = []
        if progress and st_progress:
            import streamlit as st
            st_progress.progress(0)
        # select which tqdm progress bar to show
        hiderounds = True
        hidereps = True
        if progress:
            if soft_avgs > 1:
                hiderounds = False
            else:
                hidereps = False

        # avg_d doesn't have a specific shape here, so that it's easier for child programs to write custom _average_buf
        avg_d = None
        for ir in tqdm(range(soft_avgs), disable=hiderounds):
            # Configure and enable buffer capture.
            self.config_bufs(soc, enable_avg=True, enable_buf=False)

            # Reload data memory.
            soc.reload_mem()

            count = 0
            with tqdm(total=total_count, disable=hidereps) as pbar:
                soc.start_readout(total_count, counter_addr=self.counter_addr,
                                  ch_list=list(self.ro_chs), reads_per_shot=self.reads_per_shot)
                while count < total_count:
                    new_data = obtain(soc.poll_data())
                    for new_points, (d, s) in new_data:
                        for ii, nreads in enumerate(self.reads_per_shot):
                            # print(count, new_points, nreads, d[ii].shape, total_count)
                            if new_points*nreads != d[ii].shape[0]:
                                logger.error("data size mismatch: new_points=%d, nreads=%d, data shape %s" % (
                                    new_points, nreads, d[ii].shape))
                            if count+new_points > total_count:
                                logger.error("got too much data: count=%d, new_points=%d, total_count=%d" % (
                                    count, new_points, total_count))
                            # use reshape to view the acc_buf array in a shape that matches the raw data
                            self.acc_buf[ii].reshape(
                                (-1, 2))[count*nreads:(count+new_points)*nreads] = d[ii]
                        count += new_points
                        self.stats.append(s)
                        pbar.update(new_points)
                        if progress and st_progress:
                            st_progress.progress(min(count / total_count, 1.0))
            # if we're thresholding, apply the threshold before averaging
            if threshold is None:
                d_reps = self.acc_buf
                round_d = self._average_buf(
                    d_reps, self.reads_per_shot, length_norm=True, remove_offset=remove_offset)
            else:
                d_reps = [np.zeros_like(d) for d in self.acc_buf]
                self.shots = self._apply_threshold(
                    self.acc_buf, threshold, angle, remove_offset=remove_offset)
                for i, ch_shot in enumerate(self.shots):
                    d_reps[i][..., 0] = ch_shot
                round_d = self._average_buf(
                    d_reps, self.reads_per_shot, length_norm=False)

            # sum over rounds axis
            if avg_d is None:
                avg_d = round_d
            else:
                for ii, d in enumerate(round_d):
                    avg_d[ii] += d
            if progress and st_progress:
                st_progress.progress((ir + 1) / soft_avgs)
        # divide total by rounds
        for d in avg_d:
            d /= soft_avgs

        return avg_d
