function Data = load_data(s)
%LOAD_DATA function to support the Community Detection Toolbox GUI
%
%    Reads the names of all .m files in directory s. Then, for each .m file
%    reads the name, arguments and help.
%    Results are returned in a cell array.

%validate input
if ~ischar(s)
    error('Input is not a string.');
end

%get directory contents
Data         = dir(s);

%pick the files with a .m extension
Data         = {Data(not(cellfun('isempty',strfind({Data.name},'.m')))).name};

%get the function arguments and remove the .m extension
for i = 1:length(Data)
    %open the .m file
    file  = fopen([s '/' Data{i}],'r');
    
    %read the function prototype
    fprot = fgets(file);
    
    %get the input arguments
    args  = get_args(fprot);

    %get help notes
    hlp   = help(Data{i});
    
    %remove the .m extension
    [~,name,~] = fileparts(Data{i});
    
    %save changes
    Data{i} = cell(1,3);
    Data{i}{1} = name;
    Data{i}{2} = args;
    Data{i}{3} = hlp;
    
    %close the .m file
    fclose(file);
end

end