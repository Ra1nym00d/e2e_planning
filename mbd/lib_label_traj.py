#!/usr/bin/env python3

import os
import time
import re
from cffi import FFI
import numpy as np


class GtObjTrajGen():
	def __init__(self):
		self.ffi = FFI()
		head_func = """
		/*===========================================================================*
		 * Generic type definitions: boolean_T, char_T, byte_T, int_T, uint_T,       *
		 *                           real_T, time_T, ulong_T.                        *
		 *===========================================================================*/
		typedef double real_T;
		typedef double time_T;
		typedef unsigned char boolean_T;
		typedef int int_T;
		typedef unsigned int uint_T;
		typedef unsigned long ulong_T;
		typedef char char_T;
		typedef unsigned char uchar_T;
		typedef char_T byte_T;


		/* Model entry point functions */
		extern void label_traj_calc_initialize(void);
		extern void label_traj_calc_terminate(void);

		/* Customized model step function */
		extern void label_traj_calc_step(real_T arg_target_lat_pos[100], real_T
		  arg_target_lgt_pos[100], real_T arg_ego_v[100], real_T arg_ego_w[100], real_T
		  arg_hist_x[100], real_T arg_hist_y[100]);

		"""
		self.ffi.cdef(head_func)
		self.lib     = self.ffi.dlopen("mbd/liblabel_traj_calc.so")
		self.x_t_ = self.ffi.new("double[100]")
		self.y_t_ = self.ffi.new("double[100]")
		self.lib.label_traj_calc_initialize()
		self.x_t = [0 for i in range(100)]
		self.y_t = [0 for i in range(100)]

	def update(self, obj_dx, obj_dy, ego_v, ego_w):
		self.lib.label_traj_calc_step(obj_dy, obj_dx, ego_v, ego_w, self.x_t_, self.y_t_)
		self.x_t = np.array(list(self.x_t_)) - obj_dx[0]
		self.y_t = np.array(list(self.y_t_)) - obj_dy[0]

