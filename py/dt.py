

try:
	import tables as _tb
	import numpy
except ImportError:
	from .mock_module import Mock
	_tb = Mock()
	numpy = Mock()
	_tb_success = False
else:
	_tb_success = True




from .core import Fountain, LarchError, IntStringDict
import warnings
import numbers
import collections
from .util.aster import asterize
import keyword
import pandas

class IncompatibleShape(LarchError):
	pass

class HDF5BadFormat(LarchError):
	pass

class HDF5Warning(UserWarning):
    pass


_exclusion_summary_columns = ['Data Source', 'Alternatives','# Cases Excluded','# Cases Remaining', ]



class LocalAttributeSet(object):
	_hdf5_attrs = ('CLASS','FILTERS','TITLE','VERSION')
	def __init__(self, *args, **kwargs):
		self._h5_expr = _tb.attributeset.AttributeSet(*args, **kwargs)
		self._local_expr = dict()
	def __setattr__(self, attr, value):
		if attr in ('_h5_expr','_local_expr'):
			object.__setattr__(self, attr, value)
		else:
			try:
				self._h5_expr.__setattr__(attr,value)
			except _tb.exceptions.FileModeError:
				self._local_expr[attr] = value
				warnings.warn("The HDF5 is not writable so '{}' will not be saved beyond this session.".format(attr), HDF5Warning, stacklevel=2)
	def __getattr__(self, attr):
		if attr in self._local_expr:
			return self._local_expr[attr]
		else:
			if attr in self._hdf5_attrs:
				raise AttributeError("'{}' is a HDF5 attribute and cannot used by larch.".format(attr))
			return self._h5_expr.__getattr__(attr)
	def __delattr__(self, attr):
		try:
			del self.local_expr[attr]
		except KeyError:
			pass
		self._h5_expr.__delattr__(attr)
	def __repr__(self):
		h5attr = self._h5_expr._f_list()
		loc_attr = self._local_expr.keys()
		ret = "{} attributes:".format(self._h5_expr._v__nodepath)
		for i in h5attr:
			ret += "\n  {!s}".format(i)
		for i in loc_attr:
			ret += "\n  {!s} (local only)".format(i)
		return ret
	def __contains__(self, key):
		return key in self._local_expr or (key in self._h5_expr and key not in self._hdf5_attrs)
	def __iter__(self):
		return iter(sorted(self._local_expr.keys()) + sorted(i for i in self._h5_expr._v_attrnames if i not in self._hdf5_attrs))
	def __len__(self):
		return len(self._local_expr) + len([i for i in self._h5_expr._v_attrnames if i not in self._hdf5_attrs])
	def __getitem__(self, key):
		return self.__getattr__(key)
	def __setitem__(self, key, value):
		return self.__setattr__(key,value)
	def __delitem__(self, key):
		return self.__delattr__(key)






class DT(Fountain):
	"""A wrapper for a pytables File used to get data for models.

	This object wraps a :class:`tables.File`, adding a number of methods designed
	specifically for working with choice-based data used in Larch.

	Parameters
	----------
	filename : str or None
		The filename of the HDF5/pytables to open. If None (the default) a 
		named temporary file is created to serve as the backing for an in-memory 
		HDF5 file, which is very fast as long as you've got enough
		memory to store the whole thing.
	mode : str
		The mode used to open the H5F file.  Common values are 'a' for append and 'r' 
		for read only.  See pytables for more detail.
	complevel : int
		The compression level to use for new objects created.  By default no compression
		is used, but substantial disk savings may be available by using it.
	inmemory : bool
		If True (defaults False), the H5FD_CORE driver is used and data will not in general be written
		to disk until the file is closed, when all accumulated changes will be written
		in a single batch.  This can be fast if you have sufficent memory but if an error 
		occurs all your intermediate changes can be lost.
	temp : bool
		If True (defaults False), the inmemory switch is activated and no changes will be
		written to disk when the file is closed. This is automatically set to true if
		the `filename` is None.

	.. warning::
		The normal constructor creates a :class:`DT` object linked to an existing 
		HDF5 file. Editing the object edits the file as well. 

	"""

	def clear_cached_values(self):
		try:
			del self._nCases
		except AttributeError:
			pass
		try:
			del self._nAlts
		except AttributeError:
			pass

	def _try_read_attrib(self, h5name, defaultvalue):
		attrib = "_"+h5name
		try:
			a = getattr(self.h5top._v_attrs, h5name)
		except AttributeError:
			# not available in h5, use default value and try to write that to h5
			a = defaultvalue
			try:
				setattr(self.h5top._v_attrs, h5name, a)
			except _tb.exceptions.FileModeError:
				pass
		setattr(self, attrib, a)

	def _try_write_attrib(self, h5name, value):
		setattr(self, "_"+h5name, value)
		try:
			setattr(self.h5top._v_attrs, h5name, value)
		except _tb.exceptions.FileModeError:
			pass

	def _refresh_alts(self):
		self._refresh_dna(self.alternative_names(), self.alternative_codes())

	def __init__(self, filename=None, mode='a', ipath='/larch', complevel=7, complib='zlib', h5f=None, inmemory=False, temp=False):
		if not _tb_success: raise ImportError("pytables not available")
		super().__init__()
		if isinstance(filename,str):
			import os
			filename = os.path.expanduser(filename)
		if filename is None:
			temp = True
			from .util.temporaryfile import TemporaryFile
			self._TemporaryFile = TemporaryFile(suffix='.h5f')
			filename = self._TemporaryFile.name
		if h5f is not None:
			self.h5f = h5f
			self._h5f_own = False
		else:
			kwd = {}
			if inmemory or temp:
				kwd['driver']="H5FD_CORE"
			if temp:
				kwd['driver_core_backing_store']=0
			if complevel is not None:
				kwd['filters']=_tb.Filters(complib=complib, complevel=complevel)
			self.h5f = _tb.open_file(filename, mode, **kwd)
			self._h5f_own = True
		self.source_filemode = mode
		self.source_filename = filename
		self._h5larchpath = ipath
		try:
			self.h5top = self.h5f._getOrCreatePath(ipath, True)
			#self.h5top._v_filters = _tb.Filters(complib=complib, complevel=complevel)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the larch root node at '{}' does not exist and cannot be created".format(ipath))
		try:
			self.h5idca = self.h5f._getOrCreatePath(ipath+'/idca', True)
			#self.h5idca._v_filters = _tb.Filters(complib=complib, complevel=complevel)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/idca' does not exist and cannot be created".format(ipath))
		try:
			self.h5idco = self.h5f._getOrCreatePath(ipath+'/idco', True)
			#self.h5idco._v_filters = _tb.Filters(complib=complib, complevel=complevel)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/idco' does not exist and cannot be created".format(ipath))
		try:
			self.h5alts = self.h5f._getOrCreatePath(ipath+'/alts', True)
			#self.h5alts._v_filters = _tb.Filters(complib=complib, complevel=complevel)
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/alts' does not exist and cannot be created".format(ipath))
		try:
			self.h5expr = self.get_or_create_group(self.h5top, 'expr')._v_attrs
		except _tb.exceptions.FileModeError:
			raise HDF5BadFormat("the node at '{}/expr' does not exist and cannot be created".format(ipath))
		self.expr = LocalAttributeSet(self.h5top.expr)
		self._refresh_alts()
		# helper for data access
		from .util.groupnode import GroupNode
		self.idco = GroupNode(self.h5top, 'idco')
		self.idca = GroupNode(self.h5top, 'idca')



	def __del__(self):
		if self._h5f_own:
			self.h5f.close()

	def __repr__(self):
		return "<larch.DT mode '{1}' at {0}>".format(self.source_filename, self.source_filemode)

	def create_group(self, *arg, **kwargs):
		return self.h5f.create_group(*arg, **kwargs)
	def create_array(self, *arg, **kwargs):
		return self.h5f.create_array(*arg, **kwargs)
	def create_carray(self, *arg, **kwargs):
		return self.h5f.create_carray(*arg, **kwargs)
	def create_earray(self, *arg, **kwargs):
		return self.h5f.create_earray(*arg, **kwargs)
	def create_external_link(self, *arg, **kwargs):
		return self.h5f.create_external_link(*arg, **kwargs)
	def create_hard_link(self, *arg, **kwargs):
		return self.h5f.create_hard_link(*arg, **kwargs)
	def create_soft_link(self, *arg, **kwargs):
		return self.h5f.create_soft_link(*arg, **kwargs)
	def flush(self, *arg, **kwargs):
		return self.h5f.flush(*arg, **kwargs)

	def get_or_create_group(self, where, name=None, title='', filters=None, createparents=False):
		try:
			return self.h5f.get_node(where, name=name)
		except _tb.NoSuchNodeError:
			if name is not None:
				return self.h5f.create_group(where, name, title=title, filters=filters, createparents=createparents)
			else:
				raise

	def _is_larch_array(self, where, name=None):
		n = self.h5f.get_node(where, name)
		try:
			n = n.dereference()
		except AttributeError:
			pass
		if isinstance(n, _tb.array.Array):
			return True
		if isinstance(n, _tb.group.Group):
			if '_index_' in n and '_values_' in n:
				return True
		return False

	def _is_mapped_larch_array(self, where, name=None):
		n = self.h5f.get_node(where, name)
		try:
			n = n.dereference()
		except AttributeError:
			pass
		if isinstance(n, _tb.group.Group):
			if '_index_' in n and '_values_' in n:
				return True
		return False

	def _alternative_codes(self):
		try:
			return self.h5alts.altids[:]
		except _tb.exceptions.NoSuchNodeError:
			return numpy.empty(0, dtype=numpy.int64)

	def _alternative_names(self):
		try:
			return self.h5alts.names[:]
		except _tb.exceptions.NoSuchNodeError:
			return numpy.empty(0, dtype=str)

	def _alternative_slot(self, code):
		try:
			if isinstance(code, numbers.Integral):
				return numpy.where(self._alternative_codes()==code)[0]
			elif isinstance(code, (numpy.ndarray,list,tuple) ):
				if isinstance(code[0], numbers.Integral):
					return numpy.where( numpy.in1d(self._alternative_codes(), code) )[0]
				else:
					return numpy.where( numpy.in1d(self._alternative_names(), code) )[0]
			else:
				return numpy.where(self._alternative_names()==code)[0]
		except IndexError:
			raise KeyError('code {} not found'.format(code))

	def set_alternatives(self, altids, alt_labels=None):
		try:
			self.h5f.remove_node(self.h5alts, 'altids')
		except _tb.exceptions.NoSuchNodeError:
			pass
		try:
			self.h5f.remove_node(self.h5alts, 'names')
		except _tb.exceptions.NoSuchNodeError:
			pass
		# Make new ones
		altids = numpy.asarray(altids)
		if altids.dtype != numpy.int64:
			from .util.arraytools import labels_to_unique_ids
			alt_labels, altids = labels_to_unique_ids(altids)
		h5altids = self.h5f.create_carray(self.h5alts, 'altids', obj=altids, title='elemental alternative code numbers')
		h5altnames = self.h5f.create_vlarray(self.h5alts, 'names', _tb.VLUnicodeAtom(), title='elemental alternative names')
		if alt_labels is None:
			alt_labels = ["a{}".format(a) for a in altids]
		for an in alt_labels:
			h5altnames.append( an )

	def alternative_codes(self):
		try:
			return tuple(int(i) for i in self.h5alts.altids[:])
		except _tb.exceptions.NoSuchNodeError:
			return ()

	def alternative_names(self):
		try:
			return tuple(str(i) for i in self.h5alts.names[:])
		except _tb.exceptions.NoSuchNodeError:
			return ()

	def alternative_name(self, code):
		codes = self._alternative_codes()
		idx = numpy.where(codes==code)[0][0]
		return self.h5alts.names[idx]

	def alternative_code(self, name):
		names = self._alternative_names()
		idx = numpy.where(names==name)[0][0]
		return self.h5alts.altids[idx]

	def caseids(self):
		return self.h5top.caseids[:]

	@property
	def caseindexes(self):
		return numpy.arange(int(self.h5top.caseids.shape[0]))

	def alternatives(self, format=list):
		'''The alternatives of the data.
		
		When format==list or 'list', returns a list of (code,name) tuples.
		When format==dict or 'dict', returns a dictionary with codes as keys
		and names as values
		'''
		if format==list or format=='list':
			return list(zip(self.alternative_codes(), self.alternative_names()))
		if format==dict or format=='dict':
			return {i:j for i,j in zip(self.alternative_codes(), self.alternative_names())}
		if format=='reversedict':
			return {j:i for i,j in zip(self.alternative_codes(), self.alternative_names())}
		raise TypeError('only allows list or dict')

	def nCases(self):
		if 'screen' in self.h5top:
			screen = self.get_screen_indexes()
			return int(screen.shape[0])
		else:
			return int(self.h5top.caseids.shape[0])

	def nAlts(self):
		return int(self.h5alts.altids.shape[0])


	def _remake_command(self, cmd, screen, dims):
		## Whoa nelly!
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		DOT = (OP, '.')
		COLON = (OP, ':')
		COMMA = (OP, ',')
		OBRAC = (OP, '[')
		CBRAC = (OP, ']')
		from io import BytesIO
		recommand = []
		try:
			cmd_encode = cmd.encode('utf-8')
		except AttributeError:
			cmd_encode = str(cmd).encode('utf-8')
		g = tokenize(BytesIO(cmd_encode).readline)
		screen_token = COLON if screen is None else (NAME, 'screen')
		for toknum, tokval, _, _, _  in g:
			if toknum == NAME and tokval in self.h5idca:
				if dims==1:
					raise IncompatibleShape("cannot use idca.{} in an idco expression".format(tokval))
				if self._is_mapped_larch_array(self.h5idca, tokval):
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idca'), DOT, (NAME, tokval), DOT, (NAME, '_values_'),
								OBRAC,
								(NAME, 'self'), DOT, (NAME, 'h5idca'), DOT, (NAME, tokval), DOT, (NAME, '_index_'),OBRAC,COLON,CBRAC,
								CBRAC, OBRAC, screen_token,]
					if dims>1:
						partial += [COMMA,COLON,]
					partial += [CBRAC, ]
					recommand.extend(partial)
				else:
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idca'), DOT, (NAME, tokval), OBRAC, screen_token,]
					if dims>1:
						partial += [COMMA,COLON,]
					partial += [CBRAC, ]
					recommand.extend(partial)
			elif toknum == NAME and tokval in self.h5idco:
				if self._is_mapped_larch_array(self.h5idco, tokval):
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idco'), DOT, (NAME, tokval), DOT, (NAME, '_values_'),
								OBRAC,
								(NAME, 'self'), DOT, (NAME, 'h5idco'), DOT, (NAME, tokval), DOT, (NAME, '_index_'),OBRAC,COLON,CBRAC,
								CBRAC, OBRAC,screen_token,CBRAC,]
					if dims>1:
						partial += [OBRAC,COLON,COMMA,(NAME, 'None'),CBRAC,]
					recommand.extend(partial)
				else:
					# replace NAME tokens
					partial = [ (NAME, 'self'), DOT, (NAME, 'h5idco'), DOT, (NAME, tokval), OBRAC,screen_token,CBRAC,]
					if dims>1:
						partial += [OBRAC,COLON,COMMA,(NAME, 'None'),CBRAC,]
					recommand.extend(partial)
			elif toknum == NAME and tokval in self.expr:
				partial = [ (NAME, 'self'), DOT, (NAME, 'expr'), DOT, (NAME, tokval), ]
				recommand.extend(partial)
			elif toknum == NAME and tokval in ('caseid','caseids'):
				partial = [ (NAME, 'self'), DOT, (NAME, 'h5top'), DOT, (NAME, 'caseids'), OBRAC,screen_token,CBRAC, ]
				recommand.extend(partial)
			elif toknum == NAME and tokval in ('caseindex','caseindexes'):
				partial = [ (NAME, 'self'), DOT, (NAME, 'caseindexes'), OBRAC,screen_token,CBRAC, ]
				recommand.extend(partial)
			elif toknum == NAME and tokval in ('altids',):
				partial = [ (NAME, 'self'), DOT, (NAME, 'h5alts'), DOT, (NAME, 'altids'), OBRAC,screen_token,CBRAC, ]
				recommand.extend(partial)
			else:
				recommand.append((toknum, tokval))
		ret = untokenize(recommand).decode('utf-8')
		return ret


	def process_proposed_screen(self, proposal):
		if isinstance(proposal, (list,tuple)):
			proposal = numpy.asarray(proposal)
		if isinstance(proposal, str) and proposal.casefold() in ("none","all","*"):
			n_cases = self.h5top.caseids.shape[0]
			screen = "None"
		elif (proposal is None and 'screen' not in self.h5top) or (isinstance(proposal, str) and proposal.casefold() in ("none","all","*")):
			n_cases = self.h5top.caseids.shape[0]
			screen = None
		elif isinstance(proposal, str):
			proposal = self.array_idco(proposal, screen="None", dtype=bool).squeeze()
			return self.process_proposed_screen(proposal)
		elif proposal is None:
			screen = self.get_screen_indexes()
			n_cases = screen.shape[0]
		elif isinstance(proposal, numpy.ndarray) and numpy.issubsctype(proposal.dtype, numpy.bool):
			if proposal.shape != self.h5top.caseids.shape:
				raise TypeError("Incorrect screen shape, you gave {!s} but this DT has {!s}".format(proposal.shape, self.h5top.caseids.shape))
			screen = numpy.nonzero(proposal)[0]
			n_cases = screen.shape[0]
		elif isinstance(proposal, numpy.ndarray) and numpy.issubdtype(proposal.dtype, numpy.int):
			screen = proposal
			n_cases = screen.shape[0]
		elif isinstance(proposal, int):
			screen = numpy.array([proposal], dtype=int)
			n_cases = screen.shape[0]
		else:
			raise TypeError("Incorrect screen type, you gave {!s}".format(type(proposal)))
		return screen, n_cases


	def array_idca(self, *vars, dtype=numpy.float64, screen=None, strip_nan=True):
		"""Extract a set of idca values from the DT.
		
		Generally you won't need to specify any parameters to this method beyond the
		variables to include in the array, as
		most values are determined automatically from the preset queries.
		
		Parameters
		----------
		vars : tuple of str
			A tuple giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idca` format variables.
		
		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably 
			'int64', 'float64', or 'bool'.
		screen : None or array of bool or 'None'
			If given, use this bool array to screen the caseids used to build 
			the array. If None, the default screen defined in the file is used.
			Pass the string 'None' to explicitly prevent the use of
			any screen.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,n_alts,len(vars)).
			
		"""
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite
		screen, n_cases = self.process_proposed_screen(screen)
		if isinstance(screen, str) and screen=="None":
			screen = None
		n_vars = len(vars)
		n_alts = self.nAlts()
		result = numpy.zeros([n_cases,n_alts,n_vars], dtype=dtype)
		for varnum,v in enumerate(vars):
			command = self._remake_command(v,screen,2)
			try:
				try:
					result[:,:,varnum] = eval( asterize(command) )
				except TypeError as type_err:
					if v in self.h5idca._v_children and isinstance(self.h5idca._v_children[v], _tb.Group):
						stacktuple = self.h5idca._v_children[v]._v_attrs.stack
						result[:,:,varnum] = self.array_idco(*stacktuple, screen=screen, strip_nan=strip_nan)
					else:
						raise
			except Exception as exc:
				args = exc.args
				if not args:
					arg0 = ''
				else:
					arg0 = args[0]
				arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(command)
				exc.args = (arg0,) + args[1:]
				raise
		if strip_nan:
			result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result



	def array_idco(self, *vars, dtype=numpy.float64, screen=None, strip_nan=True, explain=False):
		"""Extract a set of idco values from the DB based on preset queries.
		
		Generally you won't need to specify any parameters to this method beyond the
		variables to include in the array, as
		most values are determined automatically from the preset queries.
		However, if you need to override things for this array without changing
		the queries more permanently, you can use the input parameters to do so.
		Note that all override parameters must be called by keyword, not as positional
		arguments.
		
		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		screen : None or array of bool or 'None'
			If given, use this bool array to screen the caseids used to build 
			the array. If None, the default screen defined in the file is used.
			Pass the string 'None' to explicitly prevent the use of
			any screen.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.
		
		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably 
			numpy.int64, numpy.float64, or numpy.bool_.
		
		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite
		screen, n_cases = self.process_proposed_screen(screen)
		n_vars = len(vars)
		if isinstance(screen, str) and screen=="None":
			screen = None
		result = numpy.zeros([n_cases,n_vars], dtype=dtype)
		for varnum,v in enumerate(vars):
			command = self._remake_command(v,screen,1)
			if explain:
				print("Evaluating:",str(command))
			try:
				result[:,varnum] = eval( asterize(command) )
			except Exception as exc:
				args = exc.args
				if not args:
					arg0 = ''
				else:
					arg0 = args[0]
				arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(command)
				exc.args = (arg0,) + args[1:]
				raise
		if strip_nan:
			result = numpy.nan_to_num(result)
		from .array import Array
		result = result.view(Array)
		result.vars = vars
		return result

	def array_weight(self, *, var=None, **kwargs):
		try:
			w = self.h5idco._weight_
		except _tb.exceptions.NoSuchNodeError:
			return self.array_idco('1', **kwargs)
		else:
			return self.array_idco('_weight_', **kwargs)

	def array_choice(self, **kwargs):
		if isinstance(self.h5idca._choice_, _tb.Group):
			stacktuple = self.h5idca._choice_._v_attrs.stack
			return numpy.expand_dims(self.array_idco(*stacktuple, **kwargs), axis=-1)
		return self.array_idca('_choice_', **kwargs)

	def array_avail(self, *, var=None, dtype=numpy.bool_, **kwargs):
		try:
			av = self.h5idca._avail_
		except _tb.exceptions.NoSuchNodeError:
			return self.array_idca('1', dtype=dtype, **kwargs)
		if isinstance(self.h5idca._avail_, _tb.Group):
			stacktuple = self.h5idca._avail_._v_attrs.stack
			return numpy.expand_dims(self.array_idco(*stacktuple, dtype=dtype, **kwargs), axis=-1)
		else:
			return self.array_idca('_avail_', dtype=dtype, **kwargs)

	def get_screen_indexes(self):
		if 'screen' not in self.h5top:
			return None
		return numpy.nonzero(self.h5top.screen[:])[0]

	def set_screen(self, exclude_idco=(), exclude_idca=(), exclude_unavail=False, exclude_unchoosable=False, dynamic=False):
		"""
		Set a screen
		"""
		if 'screen' in self.h5top:
			self.h5f.remove_node(self.h5top, 'screen')
		if dynamic:
			self.h5f.create_group(self.h5top, 'screen')
		else:
			self.h5f.create_carray(self.h5top, 'screen', _tb.BoolAtom(), shape=(self.nCases(), ))
			self.h5top.screen[:] = True
		self.rescreen(exclude_idco, exclude_idca, exclude_unavail, exclude_unchoosable)


	def rescreen(self, exclude_idco=None, exclude_idca=None, exclude_unavail=None, exclude_unchoosable=None):
		"""
		Rebuild the screen based on the indicated exclusion criteria.
		
		Parameters
		----------
		exclude_idco : iterable of str
			A sequence of expressions that are evaluated as booleans using
			:meth:`DT.array_idco`. For each case, if any of these expressions
			evaluates as true then the entire case is excluded.
		exclude_idca : iterable of (altcode,str)
			A sequence of (altcode, expression) tuples, where the expression
			is evaluated as boolean using :meth:`DT.array_idca`. If the 
			expression evaluates as true for any alternative matching
			any of the codes in the altcode part of the tuple (which can be 
			an integer or an array of integers) then the case is excluded.
			Note that this excludes the whole case, not just the alternative
			in question.
		exclude_unavail : bool
			If true, then any case with no available alternatives is excluded.
		exclude_unchoosable : bool or int
			If true, then any case where an unavailable alternative is chosen 
			is excluded. Set to an integer greater than 1 to increase the 
			verbosity of the reporting.
			
		Notes
		-----
		Any method parameter can be omitted, in which case the previously used
		value of that parameter is retained.  To explicitly clear previous screens,
		pass an empty tuple for each parameter.
		"""
		if 'screen' not in self.h5top:
			raise TypeError('no screen node set, use set_screen instead')

		summary = pandas.DataFrame(columns=_exclusion_summary_columns)
		summary.index.name = "Criteria"

		def inheritable(newseq, oldseq):
			if newseq is not None:
				try:
					inherit = (len(newseq)>1 and newseq[0] == "+")
				except:
					inherit = False
				if inherit:
					if oldseq in self.h5top.screen._v_attrs:
						newseq = list(self.h5top.screen._v_attrs[oldseq]) + list(newseq[1:])
					else:
						newseq = list(newseq[1:])
				self.h5top.screen._v_attrs[oldseq] = newseq
				return newseq
			else:
				if oldseq in self.h5top.screen._v_attrs:
					return self.h5top.screen._v_attrs[oldseq]

		exclude_idco = inheritable(exclude_idco, 'exclude_idco')
		exclude_idca = inheritable(exclude_idca, 'exclude_idca')
		exclude_unavail = inheritable(exclude_unavail, 'exclude_unavail')
		exclude_unchoosable = inheritable(exclude_unchoosable, 'exclude_unchoosable')

		if isinstance(self.h5top.screen, _tb.Group):
			return

		if exclude_idco:
			startcount = 0
			ex_all = self.array_idco(*exclude_idco, screen="None")
			for j, ex in enumerate(exclude_idco):
				n = ex_all[:,:j+1].any(1).sum() - startcount
				summary.loc[ex,['# Cases Excluded', 'Data Source']] = (n, 'idco')
				startcount += n
			exclusions = ex_all.any(1)
		else:
			exclusions = self.array_idco('0', screen="None", dtype=bool)
		if exclude_idca:
			for altnum, expr in exclude_idca:
				altslot = self._alternative_slot(altnum)
				startcount = exclusions.sum()
				exclusions |= self.array_idca(expr, screen="None", dtype=bool)[:,altslot,:].any(1)
				n = exclusions.sum() - startcount
				summary.loc["All Alternatives Unavailable",['# Cases Excluded', 'Data Source']] = (n,'n/a')
		if exclude_unavail:
			startcount = exclusions.sum()
			exclusions |= (~(self.array_avail(screen="None").any(1)))
			n = exclusions.sum() - startcount
			summary.loc["All Alternatives Unavailable",['# Cases Excluded', 'Data Source']] = (n,'n/a')
			self.h5top.screen._v_attrs.exclude_result = s

		if exclude_unchoosable>1:
			n_total = 0
			excludors = numpy.logical_and(self.array_choice(screen="None", dtype=bool), ~self.array_avail(screen="None"))
			for altslot in range(self.nAlts()):
				startcount = exclusions.sum()
				exclusions |= excludors[:,altslot].squeeze()
				n = exclusions.sum() - startcount
				n_total += n
				if n:
					summary.loc["Chosen but Unavailable: {}".format(self.alternative_names()[altslot]),['# Cases Excluded', 'Data Source']] = (n,'n/a')
			if n_total==0:
				summary.loc["Chosen Alternative[s] Unavailable",['# Cases Excluded', 'Data Source']] = (0,'n/a')
		elif exclude_unchoosable:
			startcount = exclusions.sum()
			exclusions |= numpy.logical_and(self.array_choice(screen="None", dtype=bool), ~self.array_avail(screen="None")).any(1).squeeze()
			n = exclusions.sum() - startcount
			summary.loc["Chosen Alternative[s] Unavailable",['# Cases Excluded', 'Data Source']] = (n,'n/a')

		if len(summary)>0:
			summary['# Cases Remaining'][0] = self.h5top.caseids.shape[0] - summary['# Cases Excluded'][0]
		for rownumber in range(1,len(summary)):
			summary['# Cases Remaining'][rownumber] = summary['# Cases Remaining'][rownumber-1] - summary['# Cases Excluded'][rownumber]

		self.h5top.screen[:] = ~exclusions.squeeze()
		self.h5top.screen._v_attrs.exclude_result = summary


	def exclude_idco(self, expr, count=True):
		"""
		Add an exclusion factor based on idco data.
		
		This is primarily a convenience method, which calls `rescreen`.  Future 
		implementations may be modified to be more efficient.
		
		Parameters
		----------
		expr : str
			An expression to evaluate using :meth:`array_idco`, with dtype
			set to bool. Any cases that evaluate as positive are excluded from
			the dataset when provisioning.
		count : bool
			Count the number of cases impacted by adding the screen.
			
		Returns
		-------
		int
			The number of cases excluded as a result of adding this exclusion factor.
		"""
		if 'screen' not in self.h5top:
			self.set_screen()
		if count:
			startcount = self.h5top.screen[:].sum()
		self.rescreen(exclude_idco=['+', expr])
		if count:
			n = startcount - self.h5top.screen[:].sum()
#			if 'exclude_result' not in self.h5top.screen._v_attrs:
#				s = pandas.DataFrame(columns=_exclusion_summary_columns)
#				s.index.name = "Criteria"
#				self.h5top.screen._v_attrs.exclude_result = s
#			s = self.h5top.screen._v_attrs.exclude_result
#			s.loc[expr,['# Cases Excluded', 'Data Source']] = (n,'idco')
#			self.h5top.screen._v_attrs.exclude_result = s
			return n

	def exclude_idca(self, altids, expr, count=True):
		"""
		Add an exclusion factor based on idca data.

		This is primarily a convenience method, which calls `rescreen`.  Future
		implementations may be modified to be more efficient.
		
		Parameters
		----------
		altids : iterable of int
			A set of alternative to consider. Any cases for which the expression
			evaluates as positive for any of the listed altids are excluded from
			the dataset when provisioning.
		expr : str
			An expression to evaluate using :meth:`array_idca`, with dtype
			set to bool.
		count : bool
			Count the number of cases impacted by adding the screen.
		"""
		if 'screen' not in self.h5top:
			self.set_screen()
		if count:
			startcount = self.h5top.screen[:].sum()
		self.rescreen(exclude_idca=['+', (altids, expr)])
		if count:
			n = startcount - self.h5top.screen[:].sum()
#			if 'exclude_result' not in self.h5top.screen._v_attrs:
#				s = pandas.DataFrame(columns=_exclusion_summary_columns)
#				s.index.name = "Criteria"
#				self.h5top.screen._v_attrs.exclude_result = s
#			s = self.h5top.screen._v_attrs.exclude_result
#			s.loc[expr,['# Cases Excluded', 'Data Source', 'Alternatives']] = (n,'idca', altids)
#			self.h5top.screen._v_attrs.exclude_result = s
			return n

	@property
	def exclude_unchoosable(self):
		if 'exclude_unchoosable' in self.h5top.screen._v_attrs:
			return self.h5top.screen._v_attrs.exclude_unchoosable
		return False

	@exclude_unchoosable.setter
	def exclude_unchoosable(self, value):
		if not isinstance(value, (bool, int)):
			value = int(value)
		self.rescreen(exclude_idco=None, exclude_idca=None, exclude_unavail=None, exclude_unchoosable=value)

	@property
	def exclude_unavail(self):
		if 'exclude_unavail' in self.h5top.screen._v_attrs:
			return self.h5top.screen._v_attrs.exclude_unavail
		return False

	@exclude_unavail.setter
	def exclude_unavail(self, value):
		value = bool(value)
		self.rescreen(exclude_idco=None, exclude_idca=None, exclude_unavail=value, exclude_unchoosable=None)


	@property
	def exclusion_summary(self):
		'''A dataframe containing a summary of the exclusion factors.'''
		if 'exclude_result' not in self.h5top.screen._v_attrs:
			s = pandas.DataFrame(columns=_exclusion_summary_columns)
			s.index.name = "Criteria"
			self.h5top.screen._v_attrs.exclude_result = s
		ex_df = self.h5top.screen._v_attrs.exclude_result
		try:
			asfloat = ex_df['Alternatives'].values.astype(float)
		except:
			pass
		else:
			if numpy.isnan(asfloat).all():
				del ex_df['Alternatives']
		#ex_df.reset_index(inplace=True)
		return ex_df.reset_index()

	def provision(self, needs, screen=None, **kwargs):
		from . import Model
		if isinstance(needs,Model):
			m = needs
			needs = m.needs()
		else:
			m = None
		import numpy
		provide = {}
		screen, n_cases = self.process_proposed_screen(screen)
		#log = self.logger()
		log = None
		for key, req in needs.items():
			if log:
				log.info("Provisioning {} data...".format(key))
			if key=="Avail":
				provide[key] = numpy.require(self.array_avail(screen=screen), requirements='C')
			elif key=="Weight":
				provide[key] = numpy.require(self.array_weight(screen=screen), requirements='C')
			elif key=="Choice":
				provide[key] = numpy.require(self.array_choice(screen=screen), requirements='C')
			elif key[-2:]=="CA":
				provide[key] = numpy.require(self.array_idca(*req.get_variables(), screen=screen), requirements='C')
			elif key[-2:]=="CO":
				provide[key] = numpy.require(self.array_idco(*req.get_variables(), screen=screen), requirements='C')
			elif key=="Allocation":
				provide[key] = numpy.require(self.array_idco(*req.get_variables(), screen=screen), requirements='C')
		if screen is None:
			provide['caseids'] = numpy.require(self.h5top.caseids[:], requirements='C')
		else:
			provide['caseids'] = numpy.require(self.h5top.caseids[screen], requirements='C')
		if len(provide['caseids'].shape) == 1:
			provide['caseids'].shape = provide['caseids'].shape + (1,)
		if m is not None:
			return m.provision(provide)
		else:
			return provide


	def _check_ca_natural(self, column):
		if column in self.h5idca._v_leaves:
			return True
		if column in self.h5idca._v_children:
			colnode = self.h5idca._v_children[column]
			if isinstance(colnode, _tb.Group) and 'stack' in colnode._v_attrs:
				return numpy.all([self.check_co(z) for z in colnode._v_attrs.stack])

	def _check_co_natural(self, column):
		return column in self.h5idco._v_leaves

	def check_ca(self, column):
		if self._check_ca_natural(column):
			return True
		if self._check_co_natural(column):
			return True
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite
		try:
			command = self._remake_command(column,None,2)
			eval( asterize(command) )
		except:
			return False
		return True

	def multi_check_ca(self, bucket):
		"""Scan a list of string or a long string line-by-line to check if the variables are valid."""
		if isinstance(bucket, str):
			for b in bucket.split("\n"):
				ok = self.check_ca(b.strip())
				if not ok:
					raise KeyError("Data '{}' not found".format(b.strip()))
		else:
			try:
				bucket_iter = iter(bucket)
			except TypeError:
				self.multi_check_ca(str(bucket))
			else:
				for b in bucket_iter:
					self.multi_check_ca(b)

	def check_co(self, column, raise_exception=False):
		if self._check_co_natural(column):
			return True
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite
		try:
			command = self._remake_command(column,None,1)
			eval( asterize(command) )
		except:
			if raise_exception:
				raise
			return False
		return True

	def multi_check_co(self, bucket):
		"""Scan a list of string or a long string line-by-line to check if the variables are valid."""
		if isinstance(bucket, str):
			for b in bucket.split("\n"):
				ok = self.check_co(b.strip())
				if not ok:
					raise KeyError("Data '{}' not found".format(b.strip()))
		else:
			try:
				bucket_iter = iter(bucket)
			except TypeError:
				self.multi_check_co(str(bucket))
			else:
				for b in bucket_iter:
					self.multi_check_co(b)

	def variables_ca(self):
		return sorted(tuple(i for i in self.h5idca._v_children))

	def variables_co(self):
		return sorted(tuple(i for i in self.h5idco._v_children))


	def import_db(self, db, ignore_ca=('caseid','altid'), ignore_co=('caseid')):

		descrip_larch = {}
		descrip_alts = {
			'altid': _tb.Int64Col(pos=1, dflt=-999),
			'name': _tb.StringCol(itemsize=127, pos=2, dflt=""),
		}
		descrip_co = {}
		descrip_ca = {}
		vars_co = db.variables_co()
		vars_ca = db.variables_ca()
		for i in vars_co:
			if i == 'caseid':
				descrip_co[i] = _tb.Int64Col(pos=len(descrip_co), dflt=-999)
			else:
				descrip_co[i] = _tb.Float64Col(pos=len(descrip_co), dflt=numpy.nan)
		for i in vars_ca:
			if i in ('caseid','altid'):
				descrip_ca[i] = _tb.Int64Col(pos=len(descrip_ca), dflt=-999)
			else:
				descrip_ca[i] = _tb.Float64Col(pos=len(descrip_ca), dflt=numpy.nan)

		for var_ca in vars_ca:
			if var_ca not in ignore_ca:
				h5var = self.h5f.create_carray(self.h5idca, var_ca, _tb.Float64Atom(), shape=(db.nCases(), db.nAlts()), )
				arr, caseids = db.array_idca(var_ca)
				h5var[:,:] = arr.squeeze()

		for var_co in vars_co:
			if var_co not in ignore_co:
				h5var = self.h5f.create_carray(self.h5idco, var_co, _tb.Float64Atom(), shape=(db.nCases(),), )
				arr, caseids = db.array_idco(var_co)
				h5var[:] = arr.squeeze()

		h5caseids = self.h5f.create_carray(self.h5top, 'caseids', _tb.Int64Atom(), shape=(db.nCases(),), )
		h5caseids[:] = caseids.squeeze()

		h5scrn = self.h5f.create_carray(self.h5top, 'screen', _tb.BoolAtom(), shape=(db.nCases(),), )
		h5scrn[:] = True

		h5altids = self.h5f.create_carray(self.h5alts, 'altids', _tb.Int64Atom(), shape=(db.nAlts(),), title='elemental alternative code numbers')
		h5altids[:] = db.alternative_codes()

		h5altnames = self.h5f.create_vlarray(self.h5alts, 'names', _tb.VLUnicodeAtom(), title='elemental alternative names')
		for an in db.alternative_names():
			h5altnames.append( an )
		
		if isinstance(db.queries.avail, (dict, IntStringDict)):
			self.avail_idco = dict(db.queries.avail)
		else:
			h5avail = self.h5f.create_carray(self.h5idca, '_avail_', _tb.BoolAtom(), shape=(db.nCases(), db.nAlts()), )
			arr, caseids = db.array_avail()
			h5avail[:,:] = arr.squeeze()

		try:
			ch_ca = db.queries.get_choice_ca()
			self.h5f.create_soft_link(self.h5idca, '_choice_', target='/larch/idca/'+ch_ca)
		except AttributeError:
			h5ch = self.h5f.create_carray(self.h5idca, '_choice_', _tb.Float64Atom(), shape=(db.nCases(), db.nAlts()), )
			arr, caseids = db.array_choice()
			h5ch[:,:] = arr.squeeze()

		wgt = db.queries.weight
		if wgt:
			self.h5f.create_soft_link(self.h5idco, '_weight_', target='/larch/idco/'+wgt)



	@staticmethod
	def ExampleDirectory():
		'''Returns the directory location of the example data files.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. It is important that you do not edit the original data.
		'''
		import os.path
		TEST_DIR = os.path.join(os.path.split(__file__)[0],"data_warehouse")
		if not os.path.exists(TEST_DIR):
			uplevels = 0
			uppath = ""
			while uplevels < 20 and not os.path.exists(TEST_DIR):
				uppath = uppath+ ".."+os.sep
				uplevels += 1
				TEST_DIR = os.path.join(os.path.split(__file__)[0],uppath+"data_warehouse")
		if os.path.exists(TEST_DIR):
			return TEST_DIR	
		raise LarchError("cannot locate 'data_warehouse' examples directory")

	@staticmethod
	def Example(dataset='MTC', filename='{}.h5', temp=True):
		'''Generate an example data object in memory.
		
		Larch comes with a few example data sets, which are used in documentation
		and testing. This function copies the data into a HDF5 file, which you can
		freely edit without damaging the original data.
		
		Parameters
		----------
		dataset : {'MTC', 'SWISSMETRO', 'MINI', 'ITINERARY'}
			Which example dataset should be used.
		filename : str
			A filename to open the HDF5 file (even in-memory files need a name).
		temp : bool
			The example database be created in-memory; if `temp` is false,
			the file will be dumped to disk when closed.
			
		Returns
		-------
		DT
			An open connection to the HDF5 example data.
		
		'''

		import os.path
		example_h5files = {
		  'MTC':os.path.join(DT.ExampleDirectory(),"MTCWork.h5"),
		  }

		h5filters = _tb.Filters(complevel=5)

		try:
			filename_ = filename.format(dataset)
		except:
			pass
		else:
			filename = filename_

		from .util.filemanager import next_stack
		n=0
		while 1:
			try:
				tryname = next_stack(filename, plus=n, allow_natural=(n==0))
				h5f = _tb.open_file(tryname, 'w', filters=h5filters, driver="H5FD_CORE", driver_core_backing_store=0 if temp else 1)
			except ValueError:
				n += 1
				if n>1000:
					raise RuntimeError("cannot open HDF5 at {}".format(filename))
			else:
				break

		if dataset.upper() == "SWISSMETRO":
			from .util.temporaryfile import TemporaryGzipInflation
			return DT(TemporaryGzipInflation(os.path.join(DT.ExampleDirectory(),"swissmetro.h5.gz")))

		if dataset.upper() in example_h5files:

			h5f_orig = _tb.open_file(example_h5files[dataset.upper()])
			h5f_orig.get_node('/larch')._f_copy_children(h5f._getOrCreatePath("/larch", True), overwrite=True, recursive=True, createparents=False)
			self = DT(filename, 'w', h5f=h5f)
		else:

			from .db import DB
			edb = DB.Example(dataset)
			self = DT(filename, 'w', h5f=h5f)

			descrip_larch = {}
			descrip_alts = {
				'altid': _tb.Int64Col(pos=1, dflt=-999),
				'name': _tb.StringCol(itemsize=127, pos=2, dflt=""),
			}
			descrip_co = {}
			descrip_ca = {}
			vars_co = edb.variables_co()
			vars_ca = edb.variables_ca()
			for i in vars_co:
				if i == 'caseid':
					descrip_co[i] = _tb.Int64Col(pos=len(descrip_co), dflt=-999)
				else:
					descrip_co[i] = _tb.Float64Col(pos=len(descrip_co), dflt=numpy.nan)
			for i in vars_ca:
				if i in ('caseid','altid'):
					descrip_ca[i] = _tb.Int64Col(pos=len(descrip_ca), dflt=-999)
				else:
					descrip_ca[i] = _tb.Float64Col(pos=len(descrip_ca), dflt=numpy.nan)

			larchnode = h5f._getOrCreatePath("/larch", True)
			larchidca = h5f._getOrCreatePath("/larch/idca", True)
			larchidco = h5f._getOrCreatePath("/larch/idco", True)
			larchalts = h5f._getOrCreatePath("/larch/alts", True)

			for var_ca in vars_ca:
				if var_ca not in ('caseid', 'casenum', 'IDCASE' ):
					h5var = h5f.create_carray(larchidca, var_ca, _tb.Float64Atom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
					arr, caseids = edb.array_idca(var_ca)
					h5var[:,:] = arr.squeeze()

			for var_co in vars_co:
				if var_co not in ('caseid', 'casenum', 'IDCASE'):
					h5var = h5f.create_carray(larchidco, var_co, _tb.Float64Atom(), shape=(edb.nCases(),), filters=h5filters)
					arr, caseids = edb.array_idco(var_co)
					h5var[:] = arr.squeeze()

			h5caseids = h5f.create_carray(larchnode, 'caseids', _tb.Int64Atom(), shape=(edb.nCases(),), filters=h5filters)
			h5caseids[:] = caseids.squeeze()

			h5scrn = h5f.create_carray(larchnode, 'screen', _tb.BoolAtom(), shape=(edb.nCases(),), filters=h5filters)
			h5scrn[:] = True

			h5altids = h5f.create_carray(larchalts, 'altids', _tb.Int64Atom(), shape=(edb.nAlts(),), filters=h5filters, title='elemental alternative code numbers')
			h5altids[:] = edb.alternative_codes()

			h5altnames = h5f.create_vlarray(larchalts, 'names', _tb.VLUnicodeAtom(), filters=h5filters, title='elemental alternative names')
			for an in edb.alternative_names():
				h5altnames.append( an )
			
			if isinstance(edb.queries.avail, (dict, IntStringDict)):
				self.avail_idco = dict(edb.queries.avail)
			else:
				h5avail = h5f.create_carray(larchidca, '_avail_', _tb.BoolAtom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
				arr, caseids = edb.array_avail()
				h5avail[:,:] = arr.squeeze()

			try:
				ch_ca = edb.queries.get_choice_ca()
				h5f.create_soft_link(larchidca, '_choice_', target='/larch/idca/'+ch_ca)
			except AttributeError:
				h5ch = h5f.create_carray(larchidca, '_choice_', _tb.Float64Atom(), shape=(edb.nCases(), edb.nAlts()), filters=h5filters)
				arr, caseids = edb.array_choice()
				h5ch[:,:] = arr.squeeze()

			wgt = edb.queries.weight
			if wgt:
				h5f.create_soft_link(larchidco, '_weight_', target='/larch/idco/'+wgt)

		return self

	def idco_code_to_idca_dummy(self, oldvarname, newvarname, complib='zlib', complevel=5):
		'''
		Transforms an integer idco variable containing alt codes into an idca dummy variable.
		
		This method is particularly useful for cases when the choice is given in this
		kind of idco format.
		
		Parameters
		----------
		oldvarname : str
			The :ref:`idco` variable containing integer alternative codes.
		newvarname : str
			The new :ref:`idca` variable to be created.  If it already exists, it will
			be overwritten.
			
		Other Parameters
		----------------
		complib : str
			The compression library to use for the HDF5 file default filter.
		complevel : int
			The compression level to use for the HDF5 file default filter.
		'''
		choices = self.array_idco(oldvarname)
		choices = numpy.digitize(choices, self._alternative_codes(), right=True)
		ch_array = numpy.zeros((self.nCases(), len(self._alternative_codes())), dtype=numpy.float64)
		ch_array[numpy.arange(ch_array.shape[0]),choices.squeeze()] = 1
		try:
			self.h5f.remove_node(self.h5idca, newvarname)
		except _tb.exceptions.NoSuchNodeError:
			pass
		
		h5ch = self.h5f.create_carray(self.h5idca, newvarname, obj=ch_array,
					filters=_tb.Filters(complevel=complevel, complib=complib))



	@staticmethod
	def CSV_idco(filename, caseid=None, choice=None, weight=None, savename=None, alts={}, csv_args=(), csv_kwargs={}, complib='zlib', complevel=5, **kwargs):
		'''Creates a new larch DT based on an :ref:`idco` CSV data file.

		The input data file should be an :ref:`idco` data file, with the first line containing the column headings.
		The reader will attempt to determine the format (csv, tab-delimited, etc) automatically. 

		Parameters
		----------
		filename : str
			File name (absolute or relative) for CSV (or other text-based delimited) source data.
		caseid : str      
			Column name that contains the unique case id's. If the data is in idco format, case id's can
			be generated automatically based on line numbers by setting caseid to None (the default).
		choice : str or None
			Column name that contains the id of the alternative that is selected (if applicable). If not
			given, and if choice is not included in `alts` below, no _choice_ h5f node will be 
			autogenerated, and it will need to be set manually later.
		weight : str or None
			Column name of the weight for each case. If None, defaults to equal weights.
		savename : str or None
			If not None, the name of the location to save the HDF5 file that is created.
		alts : dict
			A dictionary with keys of alt codes, and values of (alt name, avail column, choice column) tuples.
			If `choice` is given, the third item in the tuple is ignored and can be omitted.
			
		Other Parameters
		----------------
		csv_args : tuple
			A tuple of positional arguments to pass to :meth:`DT.import_idco` (and by extension
			to :meth:`pandas.import_csv` or :meth:`pandas.import_excel`).
		csv_kwargs : dict
			A dictionary of keyword arguments to pass to :meth:`DT.import_idco` (and by extension
			to :meth:`pandas.import_csv` or :meth:`pandas.import_excel`).
		complib : str
			The compression library to use for the HDF5 file default filter.
		complevel : int
			The compression level to use for the HDF5 file default filter.

		Keyword arguments not listed here are passed to the :class:`DT` constructor.

		Returns
		-------
		DT
			An open :class:`DT` file.
		'''
		if len(alts)==0:
			raise ValueError('alternatives must be given for idco import (a future vresion of larch may relax this requirement)')
		
		
		self = DT(filename=savename, complevel=complevel, complib=complib, **kwargs)
		self.import_idco(filename, *csv_args, caseid_column=None, **csv_kwargs)
		
		h5filters = _tb.Filters(complevel=complevel, complib=complib)


		altscodes_seq = sorted(alts)

		h5altids = self.h5f.create_carray(self.h5alts, 'altids', _tb.Int64Atom(), shape=(len(alts),), filters=h5filters, title='elemental alternative code numbers')
		h5altids[:] = numpy.asarray(altscodes_seq)

		h5altnames = self.h5f.create_vlarray(self.h5alts, 'names', _tb.VLUnicodeAtom(), filters=h5filters, title='elemental alternative names')
		for an in altscodes_seq:
			h5altnames.append( alts[an][0] )
		# Weight
		if weight:
			self.h5f.create_soft_link(self.h5idco, '_weight_', target='/larch/idco/'+weight)
		# Choice
		try:
			self.choice_idco = {a:aa[2] for a,aa in alts.items()}
		except IndexError:
			if choice is not None:
				self.idco_code_to_idca_dummy(choice, '_choice_', complib=complib, complevel=complevel)
				self.h5idco._v_attrs.choice_indicator = choice
			else:
				raise
		# Avail
		self.avail_idco = {a:aa[1] for a,aa in alts.items()}
	
		return self

	def validate_hdf5(self, log=print, errlog=None):
		"""Generate a validation report for this DT.
		
		The generated report is fairly detailed and describes each requirement
		for a valid DT file and whether or not it is met.
		
		Parameters
		----------
		log : callable
			Typically "print", but can be replaced with a different callable 
			to accept a series of unicode strings for each line in the report.
		errlog : callable or None
			By default, None.  If not none, the report will print as with `log`
			but only if there are errors.
		
		"""
		if log is None:
			log = lambda *x: None
		nerrs = 0
		isok = None
		import textwrap
		blank_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   '     │ ',
			subsequent_indent='     │ ',
			)
		ok_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   ' ok  │ ',
			subsequent_indent='     │ ',
			)
		na_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   ' n/a │ ',
			subsequent_indent='     │ ',
			)
		errmsg_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   'ERROR│ ',
			subsequent_indent=' ┃   │ ',
			)
		errval_wrapper = textwrap.TextWrapper(
			width=80,
			tabsize=4,
			initial_indent=   ' ┣━► │ ',
			subsequent_indent=' ┣━► │ ',
			)
		def rreplace(_s, _old, _new):
			_li = _s.rsplit(_old, 1)
			return _new.join(_li)
		def zzz(message, invalid, make_na=False):
			if make_na:
				log(na_wrapper.fill(message))
			elif invalid:
				log(errmsg_wrapper.fill(message))
				if invalid is True:
					invalid_str = "Nope"
				else:
					invalid_str = str(invalid)
				log(rreplace(errval_wrapper.fill(invalid_str),'┣','┗'))
			else:
				log(ok_wrapper.fill(message))
			return 0 if (not invalid) or make_na else 1

		def category(catname):
			log('─────┼'+'─'*74)
			log(blank_wrapper.fill(catname))

		def subcategory(catname):
			log('     ├ '+'{:┄<73}'.format(catname+' '))

		## Top lines of display
		title = "{0} (with mode '{1}')".format(self.source_filename, self.source_filemode)
		#log("\u2550"*90)
		log("═"*80)
		log("larch.DT Validation for {}".format( title ))
		log("─────┬"+"─"*74)
		
		
		def isinstance_(obj, things):
			try:
				obj = obj.dereference()
			except AttributeError:
				pass
			return isinstance(obj, things)
		
		
		## TOP
		nerrs+= zzz("There should be a designated `larch` group node under which all other nodes reside.",
					not isinstance_(self.h5top, _tb.group.Group))
		
		## CASEIDS
		category('CASES')
		try:
			caseids_node = self.h5top.caseids
			caseids_nodeatom = caseids_node.atom
			caseids_node_shape = caseids_node.shape
			caseids_node_dim = len(caseids_node.shape)
			caseids_node_len = caseids_node.shape[0]
		except _tb.exceptions.NoSuchNodeError:
			caseids_node = None
			caseids_nodeatom = None
			caseids_node_shape = ()
			caseids_node_dim = 0
			caseids_node_len = 0

		nerrs+= zzz("Under the top node, there must be an array node named `caseids`.",
					'missing caseids node' if caseids_node is None else
					isok if isinstance_(caseids_node, _tb.array.Array) else
					'caseids is not an array node')

		nerrs+= zzz("The `caseids` array dtype should be Int64.",
					isok if isinstance(caseids_nodeatom, _tb.atom.Int64Atom) else "caseids dtype is {!s}".format(caseids_nodeatom))

		nerrs+= zzz("The `caseids` array should be 1 dimensional.",
					caseids_node_dim!=1)
		
		subcategory('Case Filtering')
		nerrs+= zzz("If there may be some data cases that are not to be included in the processing of "
					"the discrete choice model, there should be a node named `screen` under the top "
					"node.",
					None if 'screen' in self.h5top else None,
					'screen' not in self.h5top)

		# default failure values for screen checking
		screen_is_array = False
		screen_is_bool_array = False
		screen_shape = ()
		if 'screen' in self.h5top:
			screen_is_array = isinstance_(self.h5top.screen, _tb.array.Array)
			if screen_is_array:
				screen_is_bool_array = isinstance(self.h5top.screen.atom, _tb.atom.BoolAtom)
				screen_shape = self.h5top.screen.shape

		nerrs+= zzz("If it exists `screen` must be a Bool array.",
					not screen_is_array or not screen_is_bool_array,
					'screen' not in self.h5top)

		nerrs+= zzz("And `screen` must be have the same shape as `caseids`.",
					None if screen_shape == caseids_node_shape else "screen is {} while caseids is {}".format(screen_shape, caseids_node_shape),
					'screen' not in self.h5top
					)


		## ALTS
		category('ALTERNATIVES')
		nerrs+= zzz("Under the top node, there should be a group named `alts` to hold alternative data.",
					not isinstance_(self.h5top.alts, _tb.group.Group))
		try:
			altids_node = self.h5top.alts.altids
			altids_nodeatom = altids_node.atom
			altids_node_dim = len(altids_node.shape)
			altids_node_len = altids_node.shape[0]
		except _tb.exceptions.NoSuchNodeError:
			altids_node = None
			altids_nodeatom = None
			altids_node_dim = 0
			altids_node_len = 0
		nerrs+= zzz("Within the `alts` node, there should be an array node named `altids` to hold the "
					"identifying code numbers of the alternatives.",
					not isinstance_(altids_node, _tb.array.Array) )
		nerrs+= zzz("The `altids` array dtype should be Int64.",
					None if isinstance(altids_nodeatom, _tb.atom.Int64Atom) else "altids dtype is {!s}".format(altids_nodeatom))
		nerrs+= zzz("The `altids` array should be one dimensional.",
					None if (altids_node_dim==1) else "it has {} dimensions".format(altids_node_dim))

		try:
			altnames_node = self.h5top.alts.names
			altnames_nodeatom = altnames_node.atom
			altnames_node_len = altnames_node.shape[0]
		except _tb.exceptions.NoSuchNodeError:
			altnames_node = None
			altnames_nodeatom = None
			altnames_node_len = 0

		nerrs+= zzz("Within the `alts` node, there should also be a VLArray node named `names` to hold "
					"the names of the alternatives.",
					not isinstance_(altnames_node, _tb.vlarray.VLArray))
		nerrs+= zzz("The `names` node should hold unicode values.",
					not isinstance(altnames_nodeatom, _tb.atom.VLUnicodeAtom))
		nerrs+= zzz("The `altids` and `names` arrays should be the same length, and this will be the "
					"number of elemental alternatives represented in the data.",
					altnames_node_len!=altids_node_len)

		## IDCO
		category('IDCO FORMAT DATA')
		nerrs+= zzz("Under the top node, there should be a group named `idco` to hold that data.",
					not isinstance_(self.h5top.idco, _tb.group.Group))
		nerrs+= zzz("Every child node name in `idco` must be a valid Python identifer (i.e. starts "
					"with a letter or underscore, and only contains letters, numbers, and underscores) "
					"and not a Python reserved keyword.",
					[i for i in self.h5idco._v_children.keys() if not i.isidentifier() or keyword.iskeyword(i)])
		
		idco_child_incorrect_sized = {}
		for idco_child in self.h5idco._v_children.keys():
			if isinstance_(self.h5idco._v_children[idco_child], _tb.group.Group):
				if '_index_' not in self.h5idco._v_children[idco_child] or '_values_' not in self.h5idco._v_children[idco_child]:
					idco_child_incorrect_sized[idco_child] = 'invalid group'
			else:
				try:
					if self.h5idco._v_children[idco_child].shape[0] != caseids_node_len:
						idco_child_incorrect_sized[idco_child] = self.h5idco._v_children[idco_child].shape
				except:
					idco_child_incorrect_sized[idco_child] = 'exception'
		nerrs+= zzz("Every child node in `idco` must be (1) an array node with shape the same as `caseids`, "
					"or (2) a group node with child nodes `_index_` as an array with the correct shape and "
					"an integer dtype, and `_values_` such that _values_[_index_] reconstructs the desired "
					"data array.",
					idco_child_incorrect_sized)
		

		## WEIGHT
		subcategory('Case Weights')
		try:
			weightnode = self.h5idco._weight_
		except _tb.exceptions.NoSuchNodeError:
			weightnode = None
			weightnode_atom = None
		else:
			weightnode_atom = weightnode.atom
		nerrs+= zzz("If the cases are to have non uniform weights, then there should a `_weight_` node "
					"(or a name link to a node) within the `idco` group.",
					isok if weightnode else None,
					'_weight_' not in self.h5idco)
		nerrs+= zzz("If weights are given, they should be of Float64 dtype.",
					isok if isinstance(weightnode_atom, _tb.atom.Float64Atom) else "_weight_ dtype is {!s}".format(weightnode_atom),
					'_weight_' not in self.h5idco)


		## IDCA
		category('IDCA FORMAT DATA')
		nerrs+= zzz("Under the top node, there should be a group named `idca` to hold that data.",
					not isinstance_(self.h5top.idca, _tb.group.Group))
		nerrs+= zzz("Every child node name in `idca` must be a valid Python identifer (i.e. starts "
					"with a letter or underscore, and only contains letters, numbers, and underscores) "
					"and not a Python reserved keyword.",
					[i for i in self.h5idca._v_children.keys() if not i.isidentifier() or keyword.iskeyword(i)])

		idca_child_incorrect_sized = {}
		for idca_child in self.h5idca._v_children.keys():
			if isinstance_(self.h5idca._v_children[idca_child], _tb.group.Group):
				if '_index_' not in self.h5idca._v_children[idca_child] or '_values_' not in self.h5idca._v_children[idca_child]:
					if 'stack' not in self.h5idca._v_children[idca_child]._v_attrs:
						idca_child_incorrect_sized[idca_child] = 'invalid group'
				else:
					if self.h5idca._v_children[idca_child]._values_.shape[1] != altids_node_len:
						idca_child_incorrect_sized[idca_child] = self.h5idca._v_children[idca_child]._values_.shape
			else:
				try:
					if self.h5idca._v_children[idca_child].shape[0] != caseids_node_len or \
					   self.h5idca._v_children[idca_child].shape[1] != altids_node_len:
						idca_child_incorrect_sized[idca_child] = self.h5idca._v_children[idca_child].shape
				except:
					idca_child_incorrect_sized[idca_child] = 'exception'
		nerrs+= zzz("Every child node in `idca` must be (1) an array node with the first dimension the "
					"same as the length of `caseids`, and the second dimension the same as the length "
					"of `altids`, or (2) a group node with child nodes `_index_` as a 1-dimensional array "
					"with the same length as the length of `caseids` and "
					"an integer dtype, and a 2-dimensional `_values_` with the second dimension the same as the length "
					"of `altids`, such that _values_[_index_] reconstructs the desired "
					"data array, or (3) a group node with a `stack` attribute.",
					idca_child_incorrect_sized)

		subcategory('Alternative Availability')
		if '_avail_' in self.h5idca:
			_av_exists = True
			_av_is_array = isinstance_(self.h5idca._avail_, _tb.array.Array)
			if _av_is_array:
				_av_atom_bool = isinstance(self.h5idca._avail_.atom, _tb.atom.BoolAtom)
			else:
				_av_atom_bool = None
				try:
					_av_stack = self.h5idca._avail_._v_attrs.stack
				except:
					_av_stack = None
		else:
			_av_exists = False
			_av_is_array = None
			_av_atom_bool = None

		nerrs+= zzz("If there may be some alternatives that are unavailable in some cases, there should "
					"be a node named `_avail_` under `idca`.",
					isok if _av_exists else 'node is missing',
					not _av_exists)
		if _av_is_array:
			nerrs+= zzz("If given as an array, it should contain an appropriately sized Bool "
						"array indicating the availability status for each alternative.",
						isok if _av_is_array and _av_atom_bool else
						'not an array' if not _av_is_array else
						'not a bool array',
						not _av_exists)
		else:
			nerrs+= zzz("If given as a group, it should have an attribute named `stack` "
						"that is a tuple of `idco` expressions indicating the availability "
						"status for each alternative. The length and order of `stack` should "
						"match that of the altid array.",
						isok if _av_stack is not None and len(_av_stack)==altids_node_len else
						'no stack' if _av_stack is None else
						'stack is wrong size',
						not _av_exists)

		subcategory('Chosen Alternatives')
		if '_choice_' in self.h5idca:
			_ch_exists = True
			_ch_is_array = isinstance_(self.h5idca._choice_, _tb.array.Array)
			if _ch_is_array:
				_ch_atom_float = isinstance(self.h5idca._choice_.atom, _tb.atom.Float64Atom)
				_ch_stack = None
			else:
				_ch_atom_float = None
				try:
					_ch_stack = self.h5idca._choice_._v_attrs.stack
				except:
					_ch_stack = None
		else:
			_ch_exists = False
			_ch_is_array = None
			_ch_atom_float = None
			_ch_stack = None
		
		nerrs+= zzz("There should be a node named `_choice_` under `idca`.",
					isok if _ch_exists else 'the node is missing')
		if _ch_is_array or not _ch_exists:
			nerrs+= zzz("If given as an array, it should be a Float64 "
						"array indicating the chosen-ness for each alternative. "
						"Typically, this will take a value of 1.0 for the alternative that is "
						"chosen and 0.0 otherwise, although it is possible to have other values, "
						"including non-integer values, in some applications.",
						isok if _ch_is_array and _ch_atom_float else
						'not an array' if not _ch_is_array else
						'not a Float64 array',
						not _ch_exists)
		if not _ch_is_array or not _ch_exists:
			nerrs+= zzz("If given as a group, it should have an attribute named `stack` "
						"that is a tuple of `idco` expressions indicating the choice "
						"status for each alternative. The length and order of `stack` should "
						"match that of the altid array.",
						isok if _ch_stack is not None and len(_ch_stack)==altids_node_len else
						'no stack' if _ch_stack is None else
						'stack is wrong size',
						not _ch_exists)

		## TECHNICAL
		category('OTHER TECHNICAL DETAILS')
		nerrs+= zzz("The set of child node names within `idca` and `idco` should not overlap (i.e. "
					"there should be no node names that appear in both).",
					set(self.h5idca._v_children.keys()).intersection(self.h5idco._v_children.keys()))
		
		## Bottom line of display
		log('═════╧'+'═'*74)
		
		if errlog is not None and nerrs>0:
			self.validate_hdf5(log=errlog)
		return nerrs

	validate = validate_hdf5



	def import_idco(self, filepath_or_buffer, caseid_column=None, *args, **kwargs):
		"""Import an existing CSV or similar file in idco format into this HDF5 file.
		
		This function relies on :func:`pandas.read_csv` to read and parse the input data.
		All arguments other than those described below are passed through to that function.
		
		Parameters
		----------
		filepath_or_buffer : str or buffer
			This argument will be fed directly to the :func:`pandas.read_csv` function.
			If a string is given and the file extension is ".xlsx" then the :func:`pandas.read_excel`
			function will be used instead.
		caseid_column : None or str
			If given, this is the column of the input data file to use as caseids.  It must be 
			given if the caseids do not already exist in the HDF5 file.  If it is given and
			the caseids do already exist, a `LarchError` is raised.
		
		Raises
		------
		LarchError
			If caseids exist and are also given,
			or if the caseids are not integer values.
		"""
		import pandas
		from . import logging
		log = logging.getLogger('DT')
		log("READING %s",str(filepath_or_buffer))
		if isinstance(filepath_or_buffer, str) and filepath_or_buffer.casefold()[-5:]=='.xlsx':
			df = pandas.read_excel(filepath_or_buffer, *args, **kwargs)
		else:
			df = pandas.read_csv(filepath_or_buffer, *args, **kwargs)
		log("READING COMPLETE")
		try:
			for col in df.columns:
				log("LOADING %s",col)
				col_array = df[col].values
				try:
					tb_atom = _tb.Atom.from_dtype(col_array.dtype)
				except ValueError:
					log.warn("  column %s is not an simple compatible datatype",col)
					try:
						maxlen = int(df[col].str.len().max())
					except ValueError:
						import datetime
						if isinstance(df[col][0],datetime.time):
							log.warn("  column %s is datetime.time, converting to Time32",col)
							tb_atom = _tb.atom.Time32Atom()
							convert_datetime_time_to_epoch_seconds = lambda tm: tm.hour*3600+ tm.minute*60 + tm.second
							col_array = df[col].apply(convert_datetime_time_to_epoch_seconds).astype(numpy.int32).values
						else:
							import __main__
							__main__.err_df = df
							raise
					else:
						maxlen = max(maxlen,8)
						log.warn("  column %s, converting to S%d",col,maxlen)
						col_array = df[col].astype('S{}'.format(maxlen)).values
						tb_atom = _tb.Atom.from_dtype(col_array.dtype)
			
				if not col.isidentifier():
					log.warn("  column %s is not a valid python identifier, converting to _%s",col,col)
					col = "_"+col
				if keyword.iskeyword(col):
					log.warn("  column %s is a python keyword, converting to _%s",col,col)
					col = "_"+col
				h5var = self.h5f.create_carray(self.h5idco, col, tb_atom, shape=col_array.shape)
				h5var[:] = col_array
			if caseid_column is not None and 'caseids' in self.h5top:
				raise LarchError("caseids already exist, not setting new ones")
			if caseid_column is not None and 'caseids' not in self.h5top:
				if caseid_column not in df.columns:
					for col in df.columns:
						if col.casefold() == caseid_column.casefold():
							caseid_column = col
							break
				if caseid_column not in df.columns:
					raise LarchError("caseid_column '{}' not found in data".format(caseid_column))
				proposed_caseids_node = self.h5idco._v_children[caseid_column]
				if not isinstance(proposed_caseids_node.atom, _tb.atom.Int64Atom):
					col_array = df[caseid_column].values.astype('int64')
					if not numpy.all(col_array==df[caseid_column].values):
						raise LarchError("caseid_column '{}' does not contain only integer values".format(caseid_column))
					h5var = self.h5f.create_carray(self.h5idco, caseid_column+'_int64', _tb.Atom.from_dtype(col_array.dtype), shape=col_array.shape)
					h5var[:] = col_array
					caseid_column = caseid_column+'_int64'
					proposed_caseids_node = self.h5idco._v_children[caseid_column]
				self.h5f.create_soft_link(self.h5top, 'caseids', target=self.h5idco._v_pathname+'/'+caseid_column)
			if caseid_column is None and 'caseids' not in self.h5top:
				h5var = self.h5f.create_carray(self.h5top, 'caseids', obj=numpy.arange(1, len(df)+1, dtype=numpy.int64))
		except:
			self._df_exception = df
			raise



	def import_idca(self, filepath_or_buffer, caseid_col, altid_col, choice_col=None, force_int_as_float=True, chunksize=1e1000):
		"""Import an existing CSV or similar file in idca format into this HDF5 file.
		
		This function relies on :func:`pandas.read_csv` to read and parse the input data.
		All arguments other than those described below are passed through to that function.
		
		Parameters
		----------
		filepath_or_buffer : str or buffer
			This argument will be fed directly to the :func:`pandas.read_csv` function.
		caseid_column : None or str
			If given, this is the column of the input data file to use as caseids.  It must be 
			given if the caseids do not already exist in the HDF5 file.  If it is given and
			the caseids do already exist, a `LarchError` is raised.
		altid_col : None or str
			If given, this is the column of the input data file to use as altids.  It must be
			given if the altids do not already exist in the HDF5 file.  If it is given and
			the altids do already exist, a `LarchError` is raised.
		choice_col : None or str
			If given, use this column as the choice indicator.
		force_int_as_float : bool
			If True, data columns that appear to be integer values will still be stored as 
			double precision floats (defaults to True).
		chunksize : int
			The number of rows of the source file to read as a chunk.  Reading a giant file in moderate sized
			chunks can be much faster and less memory intensive than reading the entire file.
		
		Raises
		------
		LarchError
			Various errors.
			
		Notes
		-----
		Chunking may not work on Mac OS X due to a `known bug <http://github.com/pydata/pandas/issues/11793>`_
		in the pandas.read_csv function.
		"""
		import pandas
		casealtreader = pandas.read_csv(filepath_or_buffer, chunksize=chunksize, usecols=[caseid_col,altid_col])
		caseids = numpy.array([], dtype='int64')
		altids = numpy.array([], dtype='int64')
		for chunk in casealtreader:
			caseids = numpy.union1d(caseids, chunk[caseid_col].values)
			altids = numpy.union1d(altids, chunk[altid_col].values)

		if caseids.dtype != numpy.int64:
			from .util.arraytools import labels_to_unique_ids
			case_labels, caseids = labels_to_unique_ids(caseids)
			caseids = caseids.astype('int64')

		if 'caseids' not in self.h5top:
			self.h5f.create_carray(self.h5top, 'caseids', obj=caseids)
		else:
			if not numpy.all(caseids==self.h5top.caseids[:]):
				raise LarchError ('caseids exist but do not match the imported data')

		alt_labels = None
		if 'altids' not in self.h5alts:
			if altids.dtype != numpy.int64:
				from .util.arraytools import labels_to_unique_ids
				alt_labels, altids = labels_to_unique_ids(altids)
			h5altids = self.h5f.create_carray(self.h5alts, 'altids', obj=altids, title='elemental alternative code numbers')
		else:
			if not numpy.all(numpy.in1d(altids, self.h5alts.altids[:], True)):
				raise LarchError ('altids exist but do not match the imported data')
			else:
				altids = self.h5alts.altids[:]
		if 'names' not in self.h5alts:
			h5altnames = self.h5f.create_vlarray(self.h5alts, 'names', _tb.VLUnicodeAtom(), title='elemental alternative names')
			if alt_labels is not None:
				for an in alt_labels:
					h5altnames.append( str(an) )
			else:
				for an in self.h5alts.altids[:]:
					h5altnames.append( 'a'+str(an) )

		caseidmap = {i:n for n,i in enumerate(caseids)}
		altidmap = {i:n for n,i in enumerate(altids)}
		if alt_labels is not None:
			# if the altids are not integers, we replace the altid map with a labels map
			altidmap = {i:n for n,i in enumerate(alt_labels)}

		try:
			filepath_or_buffer.seek(0)
		except AttributeError:
			pass

		colreader = pandas.read_csv(filepath_or_buffer, nrows=1000 )
		force_float_columns = {}
		h5arr = {}
		for col in colreader.columns:
			if col in (caseid_col, altid_col): continue
			if force_int_as_float and colreader[col].dtype == numpy.int64:
				atom_dtype = _tb.atom.Float64Atom()
				force_float_columns[col] = numpy.float64
			else:
				atom_dtype = _tb.Atom.from_dtype(colreader[col].dtype)
			h5arr[col] = self.h5f.create_carray(self.h5idca, col, atom_dtype, shape=(caseids.shape[0], altids.shape[0]))
		if '_present_' not in colreader.columns:
			h5arr['_present_'] = self.h5f.create_carray(self.h5idca, '_present_', _tb.atom.BoolAtom(), shape=(caseids.shape[0], altids.shape[0]))

		try:
			filepath_or_buffer.seek(0)
		except AttributeError:
			pass

		reader = pandas.read_csv(filepath_or_buffer, chunksize=chunksize, dtype=force_float_columns, engine='c')
		try:
			for chunk in reader:
				casemap = chunk[caseid_col].map(caseidmap)
				altmap = chunk[altid_col].map(altidmap)
				for col in chunk.columns:
					if col in (caseid_col, altid_col): continue
					h5arr[col][casemap.values,altmap.values] = chunk[col].values
				if '_present_' not in chunk.columns:
					h5arr['_present_'][casemap.values,altmap.values] = True
		except:
			self._chunk = chunk
			self._casemap = casemap
			self._altmap = altmap
			self._altidmap = altidmap
			raise
		
		self.h5f.create_soft_link(self.h5idca, '_avail_', target=self.h5idca._v_pathname+'/_present_')

		if choice_col:
			if isinstance(self.h5idca._v_children[choice_col].atom, _tb.atom.Float64Atom):
				self.h5f.create_soft_link(self.h5idca, '_choice_', target=self.h5idca._v_pathname+'/'+choice_col)
			else:
				self.h5f.create_carray(self.h5idca, '_choice_', obj=self.h5idca._v_children[choice_col][:].astype('Float64'))


	def check_if_idca_is_idco(self, idca_var, return_data=False):
		if idca_var not in self.h5idca:
			raise LarchError("'{}' is not an idca variable".format(idca_var))
		arr = self.h5idca._v_children[idca_var][:]
		if '_avail_' in self.h5idca:
			av = self.h5idca._avail_[:]
			arr[~av] = numpy.nan
		result = (numpy.nanstd(arr, axis=1).sum()==0)
		if return_data:
			return result, arr, av
		return result
		
	def crack_idca(self, idca_var=None):
		if idca_var is None:
			return
		result, arr, av = self.check_if_idca_is_idco(idca_var, return_data=True)
		if result:
			newarr = numpy.nanmean(arr, axis=1)
			self.h5f.create_carray(self.h5idco, idca_var, obj=newarr)
			self.h5idca._v_children[idca_var]._f_remove()

	def new_idco(self, name, expression, dtype=numpy.float64, overwrite=False):
		"""Create a new :ref:`idco` variable.
		
		Creating a new variable in the data might be convenient in some instances.
		Although using the full expression as a data term in a model might be
		valid, the whole expression will need to be evaluated every time the data
		is loaded.  By using this method, you can evaluate the expression just once,
		and save the resulting array to the file.
		
		Note that this command does not (yet) evaluate the expression in kernel
		using the numexpr module.
		
		Parameters
		----------
		name : str
			The name of the new :ref:`idco` variable.
		expression : str
			An expression to evaluate as the new variable.
		dtype : dtype
			The dtype for the array of new data.
		overwrite : bool
			Should the variable be overwritten if it already exists, default to False.
			
		Raises
		-----
		tables.exceptions.NodeError
			If a variable of the same name already exists and overwrite is False.
		NameError
			If the expression contains a name that cannot be evaluated from within
			the existing :ref:`idco` data.
		"""
		data = self.array_idco(expression, screen="None", dtype=dtype).reshape(-1)
		if overwrite:
			self.delete_data(name)
		self.h5f.create_carray(self.h5idco, name, obj=data)

	def new_idco_from_array(self, name, arr):
		"""Create a new :ref:`idco` variable.
		
		Creating a new variable in the data might be convenient in some instances.
		If you create an array externally, you can add it to the file easily with
		this command.
		
		Parameters
		----------
		name : str
			The name of the new :ref:`idco` variable.
		arr : array
			An array to add as the new variable.  Must have the correct shape.
			
		Raises
		-----
		tables.exceptions.NodeError
			If a variable of the same name already exists.
		"""
		if self.h5top.caseids.shape != arr.shape:
			raise TypeError("new idco array must have shape {!s}".format(self.h5top.caseids.shape))
		self.h5f.create_carray(self.h5idco, name, obj=arr)

	def new_idca(self, name, expression):
		"""Create a new :ref:`idca` variable.
		
		Creating a new variable in the data might be convenient in some instances.
		Although using the full expression as a data term in a model might be
		valid, the whole expression will need to be evaluated every time the data
		is loaded.  By using this method, you can evaluate the expression just once,
		and save the resulting array to the file.
		
		Note that this command does not (yet) evaluate the expression in kernel
		using the numexpr module.
		
		Parameters
		----------
		name : str
			The name of the new :ref:`idca` variable.
		expression : str or array
			An expression to evaluate as the new variable, or an array of data.
			
		Raises
		-----
		tables.exceptions.NodeError
			If a variable of the same name already exists.
		NameError
			If the expression contains a name that cannot be evaluated from within
			the existing :ref:`idca` or :ref:`idco` data.
		"""
		if isinstance(expression, str):
			data = self.array_idca(expression, screen="None").reshape(-1)
		else:
			data = expression
		self.h5f.create_carray(self.h5idca, name, obj=data)


	def delete_data(self, name):
		"""Delete an existing :ref:`idca` or :ref:`idco` variable.
		
		If there is a variable of the same name in both idca and idco
		formats, this method will delete both.
		
		"""
		try:
			self.h5f.remove_node(self.h5idca, name)
		except _tb.exceptions.NoSuchNodeError:
			pass
		try:
			self.h5f.remove_node(self.h5idco, name)
		except _tb.exceptions.NoSuchNodeError:
			pass


	def export_idco(self, file, **formats):
		'''Export the :ref:`idco` data to a csv file.
		
		Only the :ref:`idco` table is exported, the :ref:`idca` table is ignored.  Future versions
		of Larch may provide a facility to export idco and idca data together in a 
		single idco output file.
		
		Parameters
		----------
		file : str or file-like
			If a string, this is the file name to give to the `open` command. Otherwise,
			this object is passed to :class:`csv.writer` directly.
			
		Notes
		-----
		This method uses a :class:`pandas.DataFrame` object to write the output file, using
		:meth:`pandas.DataFrame.to_csv`. Any keyword
		arguments not listed here are passed through to the writer.
		'''
		data = self.dataframe_idco(*self.variables_co(), screen="None")
		data.to_csv(file, index_label='caseid', **formats)
	


#	def set_avail_idco(self, *cols, varname='_avail_'):
#		"""Set up the :ref:`idca` _avail_ data array from :ref:`idco` variables.
#		
#		The availability array, if used, needs to be in :ref:`idca` format. If
#		your data isn't in that format, it's still easy to create the correct
#		availability array by stacking together the appropriate :ref:`idco` columns.
#		This command simplifies that process.
#		
#		Parameters
#		----------
#		cols : tuple of str
#			The names of the :ref:`idco` expressions that represent availability. 
#			They should be given in exactly the same order as they appear in the
#			alternative codes array.
#		varname : str
#			The name of the new :ref:`idca` variable to create. Defaults to '_avail_'.
#			
#		Raises
#		------
#		tables.exceptions.NodeError
#			If a variable of the name given by `varname` already exists.
#		NameError
#			If the expression contains a name that cannot be evaluated from within
#			the existing :ref:`idco` data.
#		TypeError
#			If the wrong number of cols arguments is provided; it must exactly match the
#			number of alternatives.
#			
#		Notes
#		-----
#		When the `varname` is given as '_avail_' (the default) the _avail_ node is replaced
#		with a special group node that links to the various alternatives in the :ref:`idco`
#		data, instead of copying them into a new array in the :ref:`idca` data.
#		
#		"""
#		if len(cols)==1 and len(cols[0])==self.nAlts():
#			cols = cols[0]
#		if len(cols) != self.nAlts():
#			raise TypeError('the input columns must exactly match the alternatives, you gave {} but there are {} alternatives'.format(len(cols), self.nAlts()))
#		# Raise an exception when a col is invalid
#		self.multi_check_co(cols)
#		if varname == '_avail_':
#			try:
#				self.h5f.remove_node(self.h5idca, '_avail_')
#			except _tb.exceptions.NoSuchNodeError:
#				pass
#			self.h5f.create_group(self.h5idca, '_avail_')
#			self.h5idca._avail_._v_attrs.stack = cols
#		else:
#			av = self.array_idco(*cols, dtype=numpy.bool)
#			self.new_idca(varname, av)
#			try:
#				self.h5f.remove_node(self.h5idca, '_avail_')
#			except _tb.exceptions.NoSuchNodeError:
#				pass
#			self.h5f.create_soft_link(self.h5idca, '_avail_', target=self.h5idca._v_pathname+'/'+varname)

	@property
	def avail_idco(self):
		"""The stack manager for avail data in idco format.
		
		To set a stack of idco expressions to represent availability data, 
		assign a dictionary to this attribute with keys as alternative codes
		and values as idco expressions.
		
		You can also get and assign individual alternative values using the 
		usual dictionary operations::
		
			DT.avail_idco[key]            # get expression
			DT.avail_idco[key] = value    # set expression
			
		"""
		return DT_idco_stack_manager(self, '_avail_')

	@avail_idco.setter
	def avail_idco(self, value):
		if not isinstance(value, dict):
			raise TypeError("assignment to avail_idco must be a dict")
		for k,v in value.items():
			self.avail_idco[k] = v

	@property
	def choice_idco(self):
		"""The stack manager for choice data in idco format.
		
		To set a stack of idco expressions to represent choice data,
		assign a dictionary to this attribute with keys as alternative codes
		and values as idco expressions.
		
		You can also get and assign individual alternative values using the 
		usual dictionary operations::
		
			DT.choice_idco[key]            # get expression
			DT.choice_idco[key] = value    # set expression
			
		"""
		return DT_idco_stack_manager(self, '_choice_')

	@choice_idco.setter
	def choice_idco(self, value):
		if not isinstance(value, dict):
			raise TypeError("assignment to choice_idco must be a dict")
		for k,v in value.items():
			self.choice_idco[k] = v


	def stack_idco(self, stackname, vardict=None):
		"""A stack manager for converting arbitrary data from idco to idca format.
		
		A stack is a new reference in the :ref:`idca` section of the HDF5
		file, which points to a series of columns in the :ref:`idco` section
		of the same file.  This allows for the creation of the stack without
		actually copying the data, and modifications to the :ref:`idco` data 
		will thus automatically propogate to :ref:`idca` as well.
		
		
		Parameters
		----------
		stackname : str
			A name for the :ref:`idca` variable that is created by stacking
			the various :ref:`idco` variables.
		vardict : dict
			Optionally pass a dictionary with keys as alternative codes
			and values as idco expressions, to initialize (or overwrite)
			the stack.
		
		
		Notes
		-----
		You can also get and assign individual alternative values using the
		usual dictionary operations::
		
			DT.stack_idco('newvarname')[altcode]            # get expression
			DT.stack_idco('newvarname')[altcode] = value    # set expression
			
		"""
		x = DT_idco_stack_manager(self, stackname)
		if vardict is not None:
			for k,v in vardict.items():
				x[k] = v
		return x
	

#	def set_choice_idco(self, *cols, varname='_choice_'):
#		"""Set up the :ref:`idca` _choice_ data array from :ref:`idco` variables.
#		
#		The choice array needs to be in :ref:`idca` format. If
#		your data isn't in that format, it's still easy to create the correct
#		availability array by stacking together the appropriate :ref:`idco` columns.
#		This command simplifies that process.
#		
#		Parameters
#		----------
#		cols : tuple of str
#			The names of the :ref:`idco` expressions that represent availability. 
#			They should be given in exactly the same order as they appear in the
#			alternative codes array.
#		varname : str
#			The name of the new :ref:`idca` variable to create. Defaults to '_choice_'.
#			
#		Raises
#		------
#		tables.exceptions.NodeError
#			If a variable of the name given by `varname` already exists.
#		NameError
#			If the expression contains a name that cannot be evaluated from within
#			the existing :ref:`idco` data.
#		TypeError
#			If the wrong number of cols arguments is provided; it must exactly match the
#			number of alternatives.
#		"""
#		if len(cols)==1 and len(cols[0])==self.nAlts():
#			cols = cols[0]
#		if len(cols) != self.nAlts():
#			raise TypeError('the input columns must exactly match the alternatives, you gave {} but there are {} alternatives'.format(len(cols), self.nAlts()))
#		# Raise an exception when a col is invalid
#		self.multi_check_co(cols)
#		cols = list(cols)
#		if varname == '_choice_':
#			try:
#				self.h5f.remove_node(self.h5idca, '_choice_')
#			except _tb.exceptions.NoSuchNodeError:
#				pass
#			self.h5f.create_group(self.h5idca, '_choice_')
#			self.h5idca._choice_._v_attrs.stack = cols
#		else:
#			ch = self.array_idco(*cols, dtype=numpy.float64)
#			self.new_idca(varname, ch)
#			try:
#				self.h5f.remove_node(self.h5idca, '_choice_')
#			except _tb.exceptions.NoSuchNodeError:
#				pass
#			self.h5f.create_soft_link(self.h5idca, '_choice_', target=self.h5idca._v_pathname+'/'+varname)
#


	def info(self, log=print):
		log("Variables:")
		log("  idco:")
		for i in sorted(self.variables_co()):
			log("    {}".format(i))
		log("  idca:")
		for i in sorted(self.variables_ca()):
			log("    {}".format(i))
		if len(self.expr):
			log("Expr:")
			for i in self.expr:
				log("    {}".format(i))


	@property
	def namespace(self):
		space = {}
		space.update(self.h5idco._v_children.iteritems())
		space.update(self.h5idca._v_children.iteritems())
		space.update({i:self.expr[i] for i in self.expr})
		return space

	def Expr(self, expression):
		return _tb.Expr(expression, uservars=self.namespace)


	def alogit_control_file(self, datafilename="exported_data.csv"):
		"A section of the control file that is related to data operations"
		import io
		alo = io.StringIO()
		from .util.alogit import repackage
		
		# File
		
		alo.write("file(name = {})".format(datafilename))
		alo.write("\ncaseid")
		for var in self.variables_co():
			alo.write("\n{}".format(var))

		alo.write("\n\n- end of variable list\n".format(datafilename))



		# Availability
		av_stack = self.avail_idco
		try:
			av_stack._check()
		except TypeError:
			alo.write( "- Larch Export Note:\n")
			alo.write( "-   Avail is not a stack, will be exported explicitly\n")
			alo.write( "-   The data export will need to include these columns\n")
			for anum,aname in self.alternatives():
				alo.write( "Avail(node_{0}) = avail_{0}\n".format(aname.replace(" ","_")) )
		else:
			alo.write( "- Larch Export Note:\n")
			alo.write( "-   Avail is an idco stack, will be exported normally as part of the idco data\n")
			# we can pass to alogit the same thing (repackaged)
			for anum,aname in self.alternatives():
				alo.write( "Avail(node_{0}) = {1}\n".format(aname.replace(" ","_"), repackage(av_stack[anum])) )

		# Other stacks
		for var in self.h5idca._v_children:
			if var in ('_avail_','_choice_'):
				continue
			var_ = var.replace(" ","_")
			if isinstance(self.h5idca._v_children[var], _tb.Group) and 'stack' in self.h5idca._v_children[var]._v_attrs:
				stack = self.stack_idco(var)
				alo.write("\n\n$array {}(alts)".format(var_))
				for anum,aname in self.alternatives():
					alo.write( "\n{2}(node_{0}) = {1}".format(aname.replace(" ","_"), repackage(stack[anum]), var_) )

		# Exclusions
		alo.write( "\n\n- Exclusion Factors:")
		exclude_number = 1
		if 'exclude_idco' in self.h5top.screen._v_attrs:
			for ex_co in self.h5top.screen._v_attrs.exclude_idco:
				alo.write("\nexclude({}) = {}".format(exclude_number, repackage(ex_co)))
				exclude_number += 1

		if 'exclude_idca' in self.h5top.screen._v_attrs:
			for ex_ca in self.h5top.screen._v_attrs.exclude_idca:
				raise NotImplementedError("excluding based on idca is not yet implemented for alogit export")

		# Choice
		try:
			ch_ind = self.h5idco._v_attrs.choice_indicator
		except AttributeError:
			alo.write( "\n\n- Choice: not given here\n")
		else:
			alo.write( "\n\n- Choice:")
			max_altcode = self._alternative_codes().max()
			alt_dict = collections.defaultdict(lambda:0)
			for anum,aname in zip(self._alternative_codes(), self._alternative_names()):
				alt_dict[anum] = aname
			choicelist = ",".join(  "node_{0}".format(alt_dict[i].replace(" ","_")) for i in range(1,max_altcode+1)  )
			alo.write( "\nchoice = recode({},{})".format(ch_ind,choicelist))

		return alo.getvalue()


	def seer(self):
		'''This function is experimental for now. 
		
		Generate a set of descriptive statistics (mean,stdev,mins,maxs,nonzeros,
		positives,negatives,zeros,mean of nonzero values) on the DT's idco data. 
		
		Not uses weights yet.
		'''
		
		
		from .util.xhtml import XHTML, XML_Builder
		output = XHTML('temp')
		output.title.text = "Data Summary"

		x = XML_Builder("div", {'class':"data_statistics"})

		description_catalog = {}
		from .roles import _data_description_catalog
		description_catalog.update(_data_description_catalog)

		names = self.variables_co()
		
		description_catalog_keys = list(description_catalog.keys())
		description_catalog_keys.sort(key=len, reverse=True)
		
		descriptions = numpy.asarray(names)
		
		for dnum, descr in enumerate(descriptions):
			if descr in description_catalog:
				descriptions[dnum] = description_catalog[descr]
			else:
				for key in description_catalog_keys:
					if key in descr:
						descr = descr.replace(key,description_catalog[key])
				descriptions[dnum] = descr
	
		show_descrip = (numpy.asarray(descriptions)!=numpy.asarray(names)).any()

		x.h2("idCO Data", anchor=1)


		means = []
		stdevs = []
		mins = []
		maxs = []
		nonzers = []
		posis = []
		negs = []
		zers = []
		mean_nonzer = []
		histograms = []
		
		from .util.statsummary import statistical_summary

		#means,stdevs,mins,maxs,nonzers,posis,negs,zers,mean_nonzer = self.stats_utility_co()
		for name in names:
			print("analyzing",name)
			try:
				ss = statistical_summary.compute(self.h5idco._v_children[name][:])
			except:
				means += ['err',]
				stdevs += ['err',]
				mins += ['err',]
				maxs += ['err',]
				nonzers += ['err',]
				posis += ['err',]
				negs += ['err',]
				zers += ['err',]
				mean_nonzer += ['err',]
				histograms += ['err',]
			else:
				means += [ss.mean,]
				stdevs += [ss.stdev,]
				mins += [ss.minimum,]
				maxs += [ss.maximum,]
				nonzers += [ss.n_nonzeros,]
				posis += [ss.n_positives,]
				negs += [ss.n_negatives,]
				zers += [ss.n_zeros,]
				mean_nonzer += [ss.mean_nonzero,]
				histograms += [ss.histogram,]
				
		
		ncols = 0
		stack = []
		titles = []

		if show_descrip:
			stack += [descriptions,]
			titles += ["Description",]
			ncols += 1
		else:
			stack += [names,]
			titles += ["Data",]
			ncols += 1

		ncols += 5
		stack += [means,stdevs,mins,maxs,zers,mean_nonzer]
		titles += ["Mean","Std.Dev.","Minimum","Maximum","Zeros","Mean(NonZero)"]

		try:
			use_p = (numpy.sum(posis)>0)
		except:
			use_p = True
		try:
			use_n = (numpy.sum(negs)>0)
		except:
			use_n = True

		if use_p:
			stack += [posis,]
			titles += ["Positives",]
			ncols += 1
		if use_n:
			stack += [negs,]
			titles += ["Negatives",]
			ncols += 1

		# Histograms
		stack += [histograms,]
		titles += ["Distribution",]
		ncols += 1

		if show_descrip:
			stack += [names,]
			titles += ["Data",]
			ncols += 1

		x.table
		x.thead
		x.tr
		for ti in titles:
			x.th(ti)
		x.end_tr
		x.end_thead
		try:
			with x.tbody_:
				for s in zip(*stack):
					with x.tr_:
						for thing,ti in zip(s,titles):
							if ti=="Description":
								x.td("{:s}".format(thing), {'class':'strut2'})
							elif ti=="Distribution":
								cell = x.start('td', {'class':'histogram_cell'})
								try:
									cell.append( thing )
								except TypeError:
									if isinstance(thing, str):
										cell.text = thing
									else:
										raise
								x.end('td')
							elif isinstance(thing,str):
								x.td("{:s}".format(thing))
							else:
								try:
									x.td("{:<11.7g}".format(thing))
								except TypeError:
									x.td(str(thing))
		except:
			for sn,stac in enumerate(stack):
				print(sn,stac)
			raise
		x.start('caption')
		x.data("Graphs are represented as pie charts if the data element has 4 or fewer distinct values.")
		x.simple('br')
		x.data("Graphs are orange if the zeroes are numerous and have been excluded.")
		x.end('caption')
		x.end_table

		output << x
		output.dump()
		output.view()




def _close_all_h5():
	try:
		self = _tb.file._open_files
		are_open_files = len(self._handlers) > 0
		handlers = list(self._handlers)  # make a copy
		for fileh in handlers:
			fileh.close()
	except:
		pass


import atexit as _atexit
_atexit.register(_close_all_h5)


class DT_idco_stack_manager:

	def __init__(self, parent, stacktype):
		self.parent = parent
		self.stacktype = stacktype

	def _check(self):
		def isinstance_(obj, things):
			try:
				obj = obj.dereference()
			except AttributeError:
				pass
			return isinstance(obj, things)
		if isinstance_(self.parent.h5idca._v_children[self.stacktype], _tb.Array):
			raise TypeError('The {} is an array, not a stack.'.format(self.stacktype))
		if not isinstance_(self.parent.h5idca._v_children[self.stacktype], _tb.Group):
			raise TypeError('The {} stack is not set up.'.format(self.stacktype))

	def _make_zeros(self):
		def isinstance_(obj, things):
			try:
				obj = obj.dereference()
			except AttributeError:
				pass
			return isinstance(obj, things)
		try:
			if isinstance_(self.parent.h5idca._v_children[self.stacktype], _tb.Array):
				self.parent.h5f.remove_node(self.parent.h5idca, self.stacktype)
		except (_tb.exceptions.NoSuchNodeError, KeyError):
			pass
		# create new group if it does not exist
		try:
			self.parent.h5f.create_group(self.parent.h5idca, self.stacktype)
		except _tb.exceptions.NodeError:
			pass
		if 'stack' not in self.parent.h5idca._v_children[self.stacktype]._v_attrs:
			self.parent.h5idca._v_children[self.stacktype]._v_attrs.stack = ["0"]*self.parent.nAlts()


	def __call__(self, *cols, varname=None):
		"""Set up the :ref:`idca` stack data array from :ref:`idco` variables.
		
		The choice array needs to be in :ref:`idca` format. If
		your data isn't in that format, it's still easy to create the correct
		availability array by stacking together the appropriate :ref:`idco` columns.
		This command simplifies that process.
		
		Parameters
		----------
		cols : tuple of str
			The names of the :ref:`idco` expressions that represent availability. 
			They should be given in exactly the same order as they appear in the
			alternative codes array.
		varname : str or None
			The name of the new :ref:`idca` variable to create. Defaults to None.
			
		Raises
		------
		tables.exceptions.NodeError
			If a variable of the name given by `varname` already exists.
		NameError
			If the expression contains a name that cannot be evaluated from within
			the existing :ref:`idco` data.
		TypeError
			If the wrong number of cols arguments is provided; it must exactly match the
			number of alternatives.
		"""
		if len(cols)==1 and len(cols[0])==self.parent.nAlts():
			cols = cols[0]
		if len(cols) != self.parent.nAlts():
			raise TypeError('the input columns must exactly match the alternatives, you gave {} but there are {} alternatives'.format(len(cols), self.parent.nAlts()))
		# Raise an exception when a col is invalid
		self.parent.multi_check_co(cols)
		cols = list(cols)
		if varname is None:
			try:
				self.parent.h5f.remove_node(self.parent.h5idca, self.stacktype)
			except _tb.exceptions.NoSuchNodeError:
				pass
			self.parent.h5f.create_group(self.parent.h5idca, self.stacktype)
			self.parent.h5idca._v_children[self.stacktype]._v_attrs.stack = cols
		else:
			ch = self.parent.array_idco(*cols, dtype=numpy.float64)
			self.parent.new_idca(varname, ch)
			try:
				self.parent.h5f.remove_node(self.parent.h5idca, self.stacktype)
			except _tb.exceptions.NoSuchNodeError:
				pass
			self.parent.h5f.create_soft_link(self.parent.h5idca, self.stacktype, target=self.parent.h5idca._v_pathname+'/'+varname)

	def __getitem__(self, key):
		self._check()
		slotarray = numpy.where(self.parent._alternative_codes()==key)[0]
		if len(slotarray) == 1:
			return self.parent.h5idca._v_children[self.stacktype]._v_attrs.stack[slotarray[0]]
		else:
			raise KeyError("key {} not found".format(key) )

	def __setitem__(self, key, value):
		slotarray = numpy.where(self.parent._alternative_codes()==key)[0]
		if len(slotarray) == 1:
			if self.stacktype not in self.parent.h5idca._v_children:
				self._make_zeros()
			if 'stack' not in self.parent.h5idca._v_children[self.stacktype]._v_attrs:
				self._make_zeros()
			tempobj = self.parent.h5idca._v_children[self.stacktype]._v_attrs.stack
			tempobj[slotarray[0]] = value
			self.parent.h5idca._v_children[self.stacktype]._v_attrs.stack = tempobj
		else:
			raise KeyError("key {} not found".format(key) )

	def __repr__(self):
		self._check()
		s = "<stack_idco: {}>".format(self.stacktype)
		for n,altid in enumerate(self.parent._alternative_codes()):
			s += "\n  {}: {!r}".format(altid, self[altid])
		return s
