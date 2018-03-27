function compile_WSDDN(varargin)
% COMPILE_WSDDN compile the C++/CUDA components of the WSDDN module

tokens = {vl_rootnn, 'matlab', 'mex', '.build', 'last_compile_opts.mat'} ;
last_args_path = fullfile(tokens{:}) ; opts = {} ;
if exist(last_args_path, 'file'), opts = {load(last_args_path)} ; end
opts = selectCompileOpts(opts) ;
vl_compilenn(opts{:}, varargin{:}, 'preCompileFn', @preCompileFn) ;

% -------------------------------------
function opts = selectCompileOpts(opts) 
% -------------------------------------
% really need a better fix at some point.  Oh well. C'est la vie.
keep = {'enableGpu', 'enableImreadJpeg', 'enableCudnn', 'enableDouble', ...
        'imageLibrary', 'imageLibraryCompileFlags', ...
        'imageLibraryLinkFlags', 'verbose', 'debug', 'cudaMethod', ...
        'cudaRoot', 'cudaArch', 'defCudaArch', 'cudnnRoot', 'preCompileFn'} ; 
s = opts{1} ;
f = fieldnames(s) ;
for i = 1:numel(f)
  if ~ismember(f{i}, keep)
    s = rmfield(s, f{i}) ;
  end
end
opts = {s} ;

% ------------------------------------------------------------------------------------
function [opts, mex_src, lib_src, flags] = preCompileFn(opts, mex_src, lib_src, flags)
% ------------------------------------------------------------------------------------

root = fullfile(fileparts(mfilename('fullpath')), 'matlab');
mcn_root = vl_rootnn() ;

% Build inside the module path
flags.src_dir = fullfile(root, 'src') ;
flags.mex_dir = fullfile(root, 'mex') ;
flags.bld_dir = fullfile(flags.mex_dir, '.build');
flags.cc = {} ;
if ~exist(fullfile(flags.bld_dir,'bits','impl'), 'dir')
  mkdir(fullfile(flags.bld_dir,'bits','impl')) ;
end

lib_src = {} ; mex_src = {} ;

if opts.enableGpu, ext = 'cu' ; else, ext = 'cpp' ; end

% Add the required MCN Dependencies
lib_src{end+1} = fullfile(mcn_root,'matlab','src','bits',['data.' ext]) ;
lib_src{end+1} = fullfile(mcn_root,'matlab','src','bits',['datamex.' ext]) ;
lib_src{end+1} = fullfile(mcn_root,'matlab','src','bits','impl','copy_cpu.cpp') ;
if opts.enableGpu
  lib_src{end+1} = fullfile(mcn_root,'matlab','src','bits','impl','copy_gpu.cu') ;
  lib_src{end+1} = fullfile(mcn_root,'matlab','src','bits','datacu.cu') ;
end
% flags.cc{end+1} = sprintf('-I%s', fullfile(mcn_root,'matlab','src'));
flags.base{end+1} = sprintf('-I%s', fullfile(mcn_root,'matlab','src'));

% Add module files
lib_src{end+1} = fullfile(root,'src','bits',['nnspp.' ext]) ;
mex_src{end+1} = fullfile(root,'src',['vl_nnspp.' ext]) ;
% CPU-specific files
lib_src{end+1} = fullfile(root,'src','bits','impl','spp_cpu.cpp') ;
% GPU-specific files
if opts.enableGpu
  lib_src{end+1} = fullfile(root,'src','bits','impl','spp_gpu.cu') ;
end
