function [rec,prec,ap] = VOCevaldet(image_ids,image_boxes)

% extract ground truth objects
npos=0;
for i=1:length(image_boxes)
if ~isempty(image_boxes{i}) 
        gt(i).BB= image_boxes{i};
        gt(i).det=false(0);
else
gt(i).BB  =[];
gt(i).det = [];
    end
end
npos = length(gtids)

BB =[];
confidence =[];
ids = [];
for image_id = image_ids
[boxes,scores] = detect_image(image_id);
BB=  [BB,boxes];
confidence  = [confidence,scores];
ids = [ids,image_id*ones(size(scores))]
end
% at the end: 
% BB should be a 4 x N array containing all N bounding boxes
% (4 entries are x1-left,y1-top,x2-right,y2-bottom - and y1<y2!!)
% confidence should be a 1 x N vector with the respective scores 

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=ids{d};

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end

    % assign detection as true positive/don't care/false positive
    if ovmax>=ov_threshold
           if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
                gt(i).det(jmax)=true;
            else
                fp(d)=1;            % false positive (multiple detection)
            end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

ap=VOCap(rec,prec);

if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
end
