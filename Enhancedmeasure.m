function [score]= Emeasure(FM,GT)

FM = logical(FM);
GT = logical(GT);

%Use double for computations.
dFM = double(FM);
dGT = double(GT);

%Special case:
if (sum(dGT(:))==0)% if the GT is completely black
    enhanced_matrix = 1.0 - dFM; %only calculate the black area of intersection
elseif(sum(~dGT(:))==0)%if the GT is completely white
    enhanced_matrix = dFM; %only calcualte the white area of intersection
else
    %Normal case:
    
    %1.compute alignment matrix
    align_matrix = AlignmentTerm(dFM,dGT);
    %2.compute enhanced alignment matrix
    enhanced_matrix = EnhancedAlignmentTerm(align_matrix);
end

%3.Emeasure score
[w,h] = size(GT);
score = sum(enhanced_matrix(:))./(w*h - 1 + eps);
end

% Alignment Term
function [align_Matrix] = AlignmentTerm(dFM,dGT)

%compute global mean
mu_FM = mean2(dFM);
mu_GT = mean2(dGT);

%compute the bias matrix
align_FM = dFM - mu_FM;
align_GT = dGT - mu_GT;

%compute alignment matrix
align_Matrix = 2.*(align_GT.*align_FM)./(align_GT.*align_GT + align_FM.*align_FM + eps);

end

% Enhanced Alignment Term function. f(x) = 1/4*(1 + x)^2)
function enhanced = EnhancedAlignmentTerm(align_Matrix)
enhanced = ((align_Matrix + 1).^2)/4;
end


