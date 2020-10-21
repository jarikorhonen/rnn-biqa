%-----------------------------------------------------------------------
%
%   MakeRNNFeatures.mat: make feature sequences for RNN model 
%
%   Input:  Image_files: table array of filenames for images
%           CNN_model:   CNN model used as feature extractor
%           trunc_fr:    length of full resolution sequence after 
%                        truncation (0: no truncation)
%           trunc_fr:    length of half resolution after truncation
%                        truncation (0: no truncation)
%           padding:     length after padding (0: no padding)
%   Output: feature_seq: table array of the resulting feature sequences
% 
%   Jari Korhonen, Shenzhen University, 2020
%   tested with Matlab R2020a
%

function feature_seq = MakeRNNFeatures(Image_files, ...
                                       CNN_model, ...
                                       trunc_fr, ...
                                       trunc_lr, ...
                                       padding)

    feature_seq = {};
    for i=1:length(Image_files)
        
        % Read image
        Image_files{i}
        image = cast(imread(Image_files{i}),'double');
        
        % Compute features for full resolution patches
        ftr_seq_fr = compute_image_feature_seq(image,CNN_model);
        [~,order]=sort(mean(ftr_seq_fr.^10),'descend');
        if trunc_fr > 0 && length(order) > trunc_fr
            order = order(1:trunc_fr);
        end
        ftr_seq_fr = ftr_seq_fr(:,order);
        
        % Compute features for full resolution patches
        image = imresize(image, 0.5);
        ftr_seq_lr = compute_image_feature_seq(image,CNN_model);
        if trunc_lr > 0 && length(order) > trunc_lr
            ftr_seq_lr = ftr_seq_lr(:,1:trunc_lr);
        end
        
        % Make the feature sequence table
        feature_seq{i} = [ftr_seq_lr ftr_seq_fr];
        
        % Add padding, if needed
        if padding>0 && length(feature_seq{i}(1,:))<padding
            feature_seq{i} = [zeros(length(feature_seq(:,1)), ...
                                    padding - ...
                                    length(feature_seq{i}(1,:))) ...
                              feature_seq{i}];
        end        
    end
end

%-----------------------------------------------------------------------
%
%   Compute feature sequences for individual images
%

function ftr_seq = compute_image_feature_seq(im, net)
    
    patches = extract_patches(im);
    
    % Initializations
    ftr_seq = [];
    layer = 'avg_pool';
    
    % Get features for each patch
    for i=1:length(patches(1,1,1,:))
        ftr_vec = activations(net,patches(:,:,:,i), layer, ...
                              'OutputAs','rows', ...
                              'ExecutionEnvironment','cpu')';
        ftr_seq(:,i) = ftr_vec(:);
    end 
    
    % Put vectors in appropriate order
    [~,order]=sort(mean(ftr_seq.^10),'descend');
    ftr_seq = ftr_seq(:,order);
end

%-----------------------------------------------------------------------
%
%   Extract patches from the image
%

function im_patches = extract_patches(im)

    
    % Split image in patches
    [height,width,~] = size(im);
            
    patch_size = [224 224];
            
    x_numb = ceil(width/patch_size(1));
    y_numb = ceil(height/patch_size(2));

    x_step = 1;
    y_step = 1;
    if x_numb>1 
        x_step = floor((width-patch_size(1))/(x_numb-1));
    end
    if y_numb>1
        y_step = floor((height-patch_size(2))/(y_numb-1));
    end
    
    % Loop through the image to make the patches
    im_patches = [];
    num_patches = 0;
    for i=1:x_step:width-patch_size(1)+1
        for j=1:y_step:height-patch_size(2)+1
            num_patches = num_patches + 1;
            y_range = j:j+patch_size(2)-1;
            x_range = i:i+patch_size(1)-1;
            im_patch = im(y_range, x_range,:);
            im_patches(:,:,:,num_patches) = im_patch;
        end
    end 
end

% EOF