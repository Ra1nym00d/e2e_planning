#!/usr/bin/env python3

import os
import time
import re
from cffi import FFI


class GtTrajGen():
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


		/* Customized model step function */
		extern void gt_traj_calc_step(real_T arg_vEgo[100], real_T arg_wEgo[100], real_T
		  arg_dt, real_T arg_x_p[101], real_T arg_y_p[101]);
		"""
		self.ffi.cdef(head_func)
		self.lib     = self.ffi.dlopen("./mbd/libgt_traj_calc.so")
		self.x_t = self.ffi.new("double[101]")
		self.y_t = self.ffi.new("double[101]")

	def update(self, vEgo, wEgo):
		self.lib.gt_traj_calc_step(vEgo, wEgo, 0.05, self.x_t, self.y_t)
