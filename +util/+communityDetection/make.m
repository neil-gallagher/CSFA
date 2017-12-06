function make(Grf,Alg,Cln,Eval,Opt,Var)
%MAKE automatically create clustering experiment scripts
%
%   The function is complementary to the Community Detection Toolbox GUI.
%   Experiments concering clustering tend to be similar. Therefore, one
%   needs only to specify the parameters of the experiment.
%
%   Input: 4 necessary structs and two optional
%     -Grf : info about the graph generator function
%     -Alg : information about the clustering algorithms
%     -Cln : information about the cluster number selection function
%     -Eval: information about the evaluation of the clustering function
%     -Opt : general experiment information
%     -Var : other experiment variables
%   Output: a script with the experiment
%
%   Example: (modify accordingly)
%
%  Grf = struct;
%  Alg = struct;
%  Eva = struct;
%  Opt = struct;
%
%
%  Grf.name = 'GGPlantedPartition';
%  Grf.par  = {'[0 20 40 60 80 100]','1-parameter','parameter','1'};
%
%  Alg.name = { 'GCSpectralClust1', 'GCDanon' };
%  Alg.par  = { {'A' , 'K'},...
%               {'A'} };
%  
%  Cln.name  = 'CNFixed';
%  Cln.par   = {'VV','5'};
%
%  Eva.name = 'PSJaccard'; %Jaccard Coefficient
%  Eva.par  = {'V0','V'};   %other parameters apart from the clustering derived
%
%  Opt.seed  = '10';
%  Opt.parameter = '0:0.05:0.5';
%  Opt.iters = '10';
%  Opt.filename = 'exp1';
%  Opt.status   = 'on';
%  Opt.savefig  = 'on';
%  Opt.errors   = 'on';
%  Opt.figtitle = 'Clustering on [0 20 40 60 80 100]';
%  Opt.figlegend = {'GCSpectralClust1','GCDanon'};
%  Opt.figtype   = {'b-x','r-x'};
%  Opt.figxlabel = 'p_{in} = 1 - p_{out}';
%  Opt.figylabel = 'Jaccard Coefficient';
%  Opt.figname   = 'exp1';
%
%  Var   = { {'K', '5'} ,...
%            {'V0','MakePrt([0 20 40 60 80 100])'} };
%
%  make(Grf,Alg,Cln,Eva,Opt);


if nargin < 4
    error('Not enough input arguments.');
end

if isa(Grf,'struct') == 0
    error('1st input [Grf] must be a struct');
end

if isa(Alg,'struct') == 0
    error('2nd input [Alg] must be a struct');
end

if isa(Alg,'struct') == 0
    error('3rd input [Cln] must be a struct');
end

if isa(Eval,'struct') == 0
    error('4th input [Eval] must be a struct');
end

if exist('Opt','var')
    if isa(Opt,'struct') == 0
        error('5th input [Opt] must be a struct');
    end
end

if exist('Var','var')
    if iscell(Var) == 0
        error('6th input [Var] must be a cell array');
    end
    
    if ~isempty(Var)
        vars = 1;
    else
        vars = 0;
    end
else
    vars = 0;
end

%%Default Options
nAlg       = 1;
graph      = '';
algorithm  = '';
evaluation = '';
clust_num  = '';
seed       = num2str(uint64(rand()*1000));
parameter  = '0:0.1:0.5';
iters      = '10';
status     = 0;
invert     = 0;
savefig    = 1;
errors     = 0;
filename   = 'ClustExp';
figtitle   = 'Clustering Experiment';
figlegend  = {''};
figtype    = {'b-x'};
figxlabel  = '';
figylabel  = '';
figname    = 'ClustExp';


%%Specify Parameters

%graph Parameters
if isfield(Grf,'name')
    graph = Grf.name;
else
    error('Graph function [Grf.name] not specified.');
end

if ischar(graph) == 0
    error('[Grf.name] must be a string.');
end

if isfield(Grf,'par')
    if ischar(Grf.par)
        warning('Converting [Grf.par] to cell.');
        Grf.par = {Grf.par};
    end
end

%algorithm parameters
if isfield(Alg,'name')
    algorithm = Alg.name;
else
    error('Algorithm function [Alg.name] not specified.');
end

if iscell(algorithm)
    nAlg = length(algorithm);
elseif ischar(algorithm)
    algorithm = {algorithm};
else
    error('[Alg.name] must be a string or a cell of strings.');
end

if isfield(Alg,'par')
    if length(Alg.par) ~= nAlg
        error('The number of algorithm parameters does not match the number of algorithms.');
    end
    for i = 1:nAlg
        if isempty(find(not(cellfun('isempty',strfind(Alg.par{i},'A'))), 1))
            error(['No "A" parameter for Algorithm: ',num2str(i),'. ',algorithm{i},'.']);
        end
    end
else
    warning('No parameters for the algorithm(s). Using default: "A".');
    for i = 1:nAlg
        Alg.par{i} = 'A';
    end
end

%cluster number selection parameters
if isfield(Cln,'name')
    clust_num = Cln.name;
else
    error('Cluster Number Selection function [Cln.name] not specified.');
end

if ischar(clust_num) == 0
    error('[Cln.name] must be a string.');
end

if isfield(Cln,'par')
    if ischar(Cln.par)
        warning('Converting [Cln.par] to cell.');
        Cln.par = {Cln.par};
    end
end

%evaluation parameters
if isfield(Eval,'name')
    evaluation = Eval.name;
else
    error('Evaluation function [Eval.name] not specified.');
end

if ischar(evaluation) == 0
    error('[Eval.name] must be a string.');
end

if isfield(Eval,'par')
    if ischar(Eval.par)
        warning('Converting [Eval.par] to cell.');
        Eval.par = {Eval.par};
    end
end

%option parameters
if exist('Opt','var')   
    if isfield(Opt,'seed')
        seed = Opt.seed;
    else
        seed = '100';
    end
    
    if isfield(Opt,'iters')
        if ischar(Opt.iters)
            Opt.iters = str2num(Opt.iters);
        end
        iters_tmp = floor(Opt.iters / 1);
        if iters_tmp == Opt.iters
            iters = Opt.iters;
        else
            warning(['Setting "iterations" to: ',iters_tmp]);
            iters = iters_tmp;
        end
        iters = num2str(iters);
    end
    
    if isfield(Opt,'parameter')
        parameter = Opt.parameter;
    end
    
    if isfield(Opt,'status')
        if strcmp(Opt.status,'on')
            status = 1;
        elseif strcmp(Opt.status,'off')
            status = 0;
        else
            %warning('Using default "status": off.');
        end
    end
    
    if isfield(Opt,'invert')
        if strcmp(Opt.invert,'on')
            invert = 1;
        elseif strcmp(Opt.invert,'off')
            invert = 0;
        else
            %warning('Using default "invert": off.');
        end
    end
    
    if isfield(Opt,'errors')
        if strcmp(Opt.errors,'on')
            errors = 1;
        elseif strcmp(Opt.errors,'off')
            errors = 0;
        else
            %dewarning('Using default "errors": off.');
        end
    end
    
    if isfield(Opt,'savefig')
        if strcmp(Opt.savefig,'on')
            savefig = 1;
        elseif strcmp(Opt.savefig,'off')
            savefig = 0;
        else
            warning('Using default "savefig": on.');
        end
    end
    
    if isfield(Opt,'filename')
        if ischar(Opt.filename) && ~isempty(Opt.filename)
            filename = Opt.filename;
        else
            warning(['[Opt.filename] must be a string. Using default: ',filename]);
        end
    end
    
    if isfield(Opt,'figtype')
        if iscell(Opt.figtype)
            if length(Opt.figtype) == nAlg
                figtype = Opt.figtype;
            else
                warning(['Length of [Opt.figtype] does not match number of algorithms. Using default:',figtype{:}]);
            end
        elseif ischar(Opt.figtype)          
            %find the comma positions
            cms = strfind(Opt.figtype,',');
            cms = [0 cms];
            
            %calculate the number of cells needed
            tmp = cell(length(cms),1);
            
            %set the cell values
            for i = 1:length(cms)-1
                tmp{i} = Opt.figtype(cms(i)+1:cms(i+1)-1);
            end
            tmp{end} = Opt.figtype(cms(end)+1:end);
            
            if length(tmp) == nAlg
                figtype = tmp;
            else
                warning(['Length of [Opt.figtype] does not match number of algorithms. Using default:',figtype{:}]);
            end
            
        else
            warning(['[Opt.figtype] must be a cell of strings. Using default: ',figtype{:}]);
        end
    end
    
    if isfield(Opt,'figtitle')
        figtitle = Opt.figtitle;
        if ischar(figtitle) == 0
            warning(['[Opt.figtitle] must be a string. Using default: ',figtitle]);
        end
    end
    
    if isfield(Opt,'figxlabel')
        if ischar(Opt.figxlabel)
            figxlabel = Opt.figxlabel;
        else
            warning('[Opt.figxlabel] must be a string. Using default: none');
        end
    end
    
    if isfield(Opt,'figylabel')
        if ischar(Opt.figylabel)
            figylabel = Opt.figylabel;
        else
            warning('[Opt.figylabel] must be a string. Using default: none');
        end
    end
    
    if isfield(Opt,'figlegend')
        if iscell(Opt.figlegend)
            figlegend = Opt.figlegend;
        elseif ischar(Opt.figlegend)
            figlegend = {Opt.figlegend};
        else
            warning('[Opt.figlegend] must be a string or a cell of strings. Using default: none');
        end
    end
    
    if isfield(Opt,'figname')
        if ischar(Opt.figname)
            figname = Opt.figname;
        else
            warning(['[Opt.figylabel] must be a string. Using default: ',figname]);
        end
    end
end

if exist('Var','var')
    %check if Var is of proper format
    if  not(all(eq(cellfun('length',Var),2)))
        vars = 0;
        warning('[Var] is not of proper format. No variables are used.');
    else
        %if there is a non-string argument
        if any(cell2mat(cellfun(@(x) cellfun(@ischar,x),Var,'UniformOutput',false))==0)
            vars = 0;
            warning('[Var] should contain only strings. No variables are used.');
        end
    end
end
        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Write experiment
file = fopen(filename,'w');

%description
tm = uint64(clock);
fprintf(file,'%s\n',['%%...Clustering Experiment...']);
fprintf(file,'%s\n',['%...',num2str(tm(1)),'.',num2str(tm(2)),'.',num2str(tm(3)),'-',num2str(tm(4)),':',num2str(tm(5)),':',num2str(tm(6)),'...']);
fprintf(file,'%s\n%s\n','%% Graph Function: ',['%     ',graph]);
fprintf(file,'%s\n','%% Algorithms:');
for i = 1:nAlg
    fprintf(file,'%s\n',['%    ',num2str(i),'. ',algorithm{i}]);
end
fprintf(file,'%s\n%s\n','%% Cluster Number Selection: ',['%     ',clust_num]);
fprintf(file,'%s\n%s\n%s\n\n','%% Evaluation: ',['%     ',evaluation],'%%');

%empty all
%fprintf(file,'%s\n%s\n\n','%%empty all', 'clear all; clc;');

%set seed
fprintf(file,'%s\n%s\n\n','%%set seed', ['set_seed(',seed,');']);

%loop options
fprintf(file,'%s\n%s\n%s\n\n','%%loop options', ['iters = ',iters,';'], ['parameter = ',parameter,';']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%general variables
if vars || invert || errors
    fprintf(file,'\n%s\n','%%general variables');
end
if vars
    %{
    for i = 1:length(Opt.var)
        fprintf(file,'%s\n',[Opt.var{i}{1}, ' = ',Opt.var{i}{2},';']);
    end
    %}
    for i = 1:length(Var)
        fprintf(file,'%s\n', [Var{i}{1} ' = ' Var{i}{2} ';']);
    end
end
if invert
    fprintf(file,'%s\n','%p = randperm();');
end
if errors
    fprintf(file,'%s\n%s\n',['nAlgs = ',num2str(nAlg),';'],'err = 0;');
end

%result variables
fprintf(file,'\n%s\n','%%result variables');
for i = 1:nAlg
    fprintf(file,'%s\n%s\n\n',['a',num2str(i),' = zeros(length(parameter),1);'], ['tmpa',num2str(i),' = zeros(iters,1);']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%Main loop
fprintf(file,'%s\n%s\n','%%main loop', 'for i = 1:length(parameter)');
fprintf(file,'  for j = 1:iters\n');

%make graph
if any(cell2mat(cellfun(@(x) strcmp(x{1},'V0'),Var,'UniformOutput',false))) %variable V0 exists; we should have it stored
    fprintf(file,'%s',['    A = ',graph,'(']);
else
    fprintf(file,'%s',['    [A, V0] = ',graph,'(']);
end
for i = 1:(length(Grf.par)-1)
      d = strfind(Grf.par{i},'parameter');
      if isempty(d) == 0
         for j = 1:length(d)
             k = d(j);
             Grf.par{i} = [Grf.par{i}(1:(k+8)), '(i)',Grf.par{i}((k+9):end)];
             d = d + 3; %the (i) term was addded
         end
      end
      fprintf(file,'%s',[Grf.par{i},',']);
end
d = strfind(Grf.par{end},'parameter');
if isempty(d) == 0
   for j = 1:length(d)
       k = d(j);
       Grf.par{end} = [Grf.par{end}(1:(k+8)), '(i)',Grf.par{end}((k+9):end)];
       d = d + 3; %the (i) term was addded
   end
end
fprintf(file,'%s\n\n',[Grf.par{end},');']);

%invert graph
if invert
    fprintf(file,'%s\n','    %A = permGrf(A,p,''do'');');
end
for i = 1:nAlg
    %errors
    if errors
        fprintf(file,'%s\n%s','    try','  ');
    end
    
    %cluster
    fprintf(file,'%s',['    VV = ',algorithm{i},'(']);
    for j = 1:(length(Alg.par{i})-1)
        fprintf(file,'%s',[Alg.par{i}{j},',']);
    end
    fprintf(file,'%s\n',[Alg.par{i}{end},');']);
    
    
    
    %cluster number variable
    fprintf(file,'%s', ['    CN = ',clust_num,'(']);
    for j = 1:(length(Cln.par)-1)
        fprintf(file,'%s',[Cln.par{j},',']);
    end
    fprintf(file,'%s\n',[Cln.par{end},');']);
    
    %correct cluster selection - (maybe save code in seperate function?)
    fprintf(file,'\n%s\n','    %%choose the cluster number');
    %fprintf(file,'%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n\n','    q = 1;','    VV_size = size(VV);',...
    %             '    for l = 1:VV_size(2)',...
    %             '       if CN == length(unique(VV(:,l)))','          q = l;',...
    %             '          break;','       end','    end','    V = VV(:,q);');
    fprintf(file,'%s\n\n','    V = VV(:,CN);');
    
    %evaluate
    if errors
        fprintf(file,'%s','  ');
    end
    fprintf(file,'%s',['    tmpa',num2str(i),'(j) = ',evaluation,'(']);
    for j = 1:(length(Eval.par)-1)
        fprintf(file,'%s',[Eval.par{j},',']);
    end
    fprintf(file,'%s\n',[Eval.par{end},');']);
    if ~errors
        fprintf(file,'\n');
    end
    
    %errors
    if errors
        fprintf(file,'%s\n%s\n%s\n%s\n\n','    catch','      err = err + 1;',['      tmpa',num2str(i),'(j) = NaN;'],'    end');
    end
end
   
fprintf(file,'%s\n','  end');

%mean of the results
for i = 1:nAlg
    if errors
        fprintf(file,'%s\n',['a',num2str(i),'(i) = nanmean(tmpa',num2str(i),');']);
    else
        fprintf(file,'%s\n',['a',num2str(i),'(i) = mean(tmpa',num2str(i),');']);
    end
end

%print status
if status
    fprintf(file,'\n%s\n','print_status(i,length(parameter));');
end
fprintf(file,'%s\n\n','end');

if errors
    fprintf(file,'%s\n%s\n%s\n\n','if err > 0','  fprintf(''Error percent: %.2f %%\n'',err/(length(parameter)*iters*nAlgs)*100);','end');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Plotting
fprintf(file,'%s\n','%%plotting');
%plotting basics
fprintf(file,'%s\n','h = figure;');
if length(figtype) == nAlg
        for i = 1:nAlg
            fprintf(file,'%s\n',['plot(parameter,a',num2str(i),',''',figtype{i},'''); hold on;']);
        end
else
      for i = 1:nAlg
            fprintf(file,'%s\n',['plot(parameter,a',num2str(i),',''',figtype{1},'''); hold on;']);
      end 
end
fprintf(file,'%s\n','hold off;');

%plotting options
fprintf(file,'\n%s\n',['title(''',figtitle,''');']);
if isempty(figxlabel) == 0
    fprintf(file,'%s\n',['xlabel(''',figxlabel,''');']);
end
if isempty(figylabel) == 0
    fprintf(file,'%s\n',['ylabel(''',figylabel,''');']);
end
if not(length(figlegend) == 1 && isempty(figlegend{1}))
    fprintf(file,'%s','legend(');
    for i = 1:(length(figlegend)-1);
        fprintf(file,'%s',['''',figlegend{i},''',']);
    end
    fprintf(file,'%s\n',['''',figlegend{end},''');']);
end

if savefig
    fprintf(file,'%s\n',['saveas(gcf,''',figname,'.fig'',','''fig'');']);
    fprintf(file,'%s\n',['saveas(gcf,''',figname,'.jpg'',','''jpg'');']);
end

end