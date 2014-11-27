function [input_image,points] =  load_im(image_id,label,normalize,point)
if nargin==2,
    normalize = 1; point = [];
end

this_file  = mfilename('fullpath'); 
[dir_up,~] = fileparts(fileparts(this_file)); %% go one directory up  
data_dir   = fullfile(dir_up,'data');         %% and then move into data  
if label==1,
    %% load face image 
    face_name        = sprintf('BioID_%i%i%i%i',ths(image_id),hun(image_id),dec(image_id),un(image_id));
    data_dir         = fullfile(data_dir,'faces');
    fnm_ground_truth = fullfile(data_dir,'points_20',[face_name,'.pts']);
    fnm_load         = fullfile(data_dir,[face_name,'.pgm']);
    
    face_image  = pgmread3_local(fnm_load)/256;
    row_min     = min(face_image,[],2);
    row_min     = find(row_min>0,1,'last');
    face_image  = face_image(1:row_min,:);
    points      = scan_points(fnm_ground_truth);
    face_image = face_image/256;
    
    if normalize
        horizontal_distance_eyes = points(1,2) - points(1,1);
        nominal_distance = 80;
        
        
        resize_factor =  horizontal_distance_eyes/nominal_distance;
        [size_x,size_y] = size(face_image);
        size_x_new      = 2*div(size_x/resize_factor,2);
        size_y_new      = 2*div(size_y/resize_factor,2);
        resize_factor_x = size_x/size_x_new;
        resize_factor_y = size_y/size_y_new;
        face_image      = imresize(face_image,[size_x_new,size_y_new],'bilinear');
        points(1,:)     = points(1,:)/resize_factor_x;
        points(2,:)     = points(2,:)/resize_factor_y;
        
        face_image      = max(face_image,0);
        face_image      = face_image/max(face_image(:));
    end
    
    input_image = face_image;
    points =  round(points);
    points = points(:,[1:5]);
    if nargin==4,
        points = points(:,point);
    end
else
    %% load background image
    fnm_load    = fullfile(data_dir,'back',sprintf('%i.jpg',image_id));
    input_image = double(rgb2gray(imread(fnm_load)))/255;
    [sv,sh]     = size(input_image);
    offset      = 20;
    step        = 20;
    cr_v        = [offset:step:sv-offset];
    cr_h        = [offset:step:sh-offset];
    [crh,crv]   = meshgrid(cr_h,cr_v);
    points      = [crh(:)';crv(:)'];   
end

    
function r = ths(input);
r = floor(input/1000);
if r>9,
    r = un(r);
end

function r = hun(input);
r = floor(input/100);
if r>9,
    r = un(r);
end
function r = un(i);
r = i - 10*floor(i/10);

function r = dec(input);
r = floor(input/10);
if r>9,
    r = r - 10*floor(r/10);
end

function [positions_full] = scan_points(filename)
fio = fopen(filename,'r');
for k=1:3, 
    fgetl(fio);  
end
for k=1:20
    line = fgetl(fio); 
    positions(:,k) = sscanf(line,'%f %f'); 
    
end 
fclose(fio);
positions_full  = positions(:,[1:4,15,5:14,16:end]);

function [X]= pgmread3_local(filename)
%PGMREAD Read a PGM (Portable Gray Map) file from disk. Only binary 
%	encoded PGM images ((P5)are supported.  
%       [X]=PGMREAD('filename') reads the file 'filename' and returns
%       the indexed image X. If no extension is given for the filename,
%	the extension '.pgm' is assumed.
%
%       See also: PGMWRITE, BPMREAD, GIFREAD, HDFREAD, PCXREAD, 
%		  TIFFREAD, XWDREAD.
%
%       Marcelo Neira Eid  12/13/96
%	mne@puc.cl
%       Last revision: Mon Dec  9 15:24:45 PST 1996
 

if (nargin~=1)
        error('Requires a filename as an argumenty.');
end;
if (isstr(filename)~=1)
        error('Requires a string filename as an argument.');
end;
if (isempty(findstr(filename,'.'))==1)
        filename=[filename,'.pgm'];
end;

fid=fopen(filename,'rb');
if (fid==-1)
        error(['Error opening ',filename,' for input.']);
end;

aux=fgetl(fid);
if (strcmp(aux,'P5')==0)
fclose(fid)
error([filename, ' is not a valid PGM binary encoded image']);
end;

% Below the comments are stripped
comments=1;
while(comments)
aux=fgets(fid);
if (aux(1)~='#')
comments=0;
end;
end;

% Get the dimensions
%[width height]=strtok(aux);
width=str2num(aux);
aux=fgets(fid);
height=str2num(aux);

%height=str2num(height);

% This strip the number of grays information. Since we know they are 255
% a priori, there isn't need to capture this information
aux=fgets(fid);
X=fread(fid);
fclose(fid);
X=reshape(X,width,height);

% The image is transposed after the read. Also Matlab pixels
% start from 1 so we transpose the image and add 1 to it. 

X=X'+1;

function res = div(a,b)
res = floor(a/b);


