function saveToJson(saveFile)

data = load(saveFile);

% convert CSFA models to structs for saving
% remove fields that can't be saved to json or take up too much space
warning('off','MATLAB:structOnObject')
for m = 1:numel(data.csfaModels)
  if ~isfield(data.csfaModels{m}, 'trainModels'), continue, end
  for t = 1:numel(data.csfaModels{m}.trainModels)
    tempStruct = struct(data.csfaModels{m}.trainModels(t));
    tempStruct = rmfield(tempStruct, 'classModel');
    tempStruct = rmfield(tempStruct, 'group');
    tempStruct.kernel = rmfield(struct(tempStruct.kernel), 'scores');
    modelStructs(t) = tempStruct;

    if isfield(data.csfaModels{m}, 'holdoutModels')
      % repeat for validation models
      tempStruct2 = struct(data.csfaModels{m}.holdoutModels(t));
      modelStructs2(t) = rmfield(tempStruct2, 'scores');
    end
  end
  data.csfaModels{m}.trainModels = modelStructs;
  data.csfaModels{m}.trainOpts = rmfield(data.csfaModels{m}.trainOpts, 'algorithm');

  % convert scores to single for more efficient json conversion
  % consider removing scores entirely and loading from .mat file when needed
  data.csfaModels{m}.scores = single(data.csfaModels{m}.scores);
end
warning('on','MATLAB:structOnObject')

jData = jsonencode(data);
saveJsonFile=strrep(saveFile,'.mat','.json');
% Create a new file and print json-encoded data to it
filename = sprintf(saveJsonFile);
fID = fopen(filename,'w+');
fprintf(fID,'%s',jData);

end
