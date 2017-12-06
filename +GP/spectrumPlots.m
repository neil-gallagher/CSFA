classdef spectrumPlots < handle
    % spectrumPlots   Plotting methods for multi-output GPs.
    %
    % spectrumPlots Methods:
    %    plotCsd - Plot all cross-spectra
    %    plotCsdComp - Plot single cross-spectrum component
    
    methods(Static)
        
        function plotCsd(s,UKU,varargin)
            % PLOTCSD   Plot the cross-spectral density (all cross-phase
            % and cross-amplitude spectra).
            %    PLOTCSD(s,UKU) plots the full cross spectrum at the
            %    frequency locations indicated in the N-dimensional vector
            %    s.  The input UKU is a C-by-C-by-N array, holding the
            %    block diagonal elements of the discrete Fourier transform
            %    of the covariance matrix, U*K*U', where U is the unitary
            %    DFT matrix.
            %
            %    PLOTCSD(s,UKU,'param1',value1,'param2',value2,...) plots
            %    the full cross spectrum with the specified parameters.
            %    Note that the param-value pairs may be included in any
            %    order.
            %
            %    Parameters:
            %       minFreq         - Minimum frequency (default 0)
            %       maxFreq         - Maximum frequency (default 30)
            %       logAmplitude    - Plot in log-space (default false)
            %       minAmplitude    - Minimum amplitude (default 1e-4)
            %       maxAmplitude    - Maximum amplitude (default data max)
            %       names           - Cell array of channel names
            %       vertLines       - Array of frequency locations to draw
            %                         dashed vertical lines (default [4 8
            %                         13 30] to reflect brain wave bands)
            %       zeroLine        - Show horizontal line at zero phase
            %                         (default true)
            %       showAmplitude   - Show amplitude tick labels on plot
            %
            %    Examples:
            %       PLOTCSD(s,UKU,'minFreq',5,'maxFreq',10)
            %    will plot the cross-spectral density between 5 and 10 Hz.
            
            C = size(UKU,1);
            
            % parse variable number of input arguments into opts
            p = GP.spectrumPlots.csdInputParser();
            addParameter(p,'names',num2cell(num2str((1:C)')),@iscell)
            addParameter(p,'showAmplitude',true,@islogical);
            parse(p,s,UKU,varargin{:});
            opts = p.Results;
            
            opts.names = cellfun(@(x)strrep(x,'_',''),opts.names,'un',0);
            
            ha = util.tight_subplot(C,C,[.01 .01],[.1 .05],[.2 .2]);
            
            allAmplitudes = abs(UKU);
            allPhases = wrapToPi(angle(UKU));
            
            if isnan(opts.maxAmplitude)
                opts.maxAmplitude = max(allAmplitudes(:));
            end
            
            opts.minAmplitude = GP.spectrumPlots.calculateMinAmplitude(opts,...
                allAmplitudes);
            
            for c2 = 1:C
                for c1 = c2:C
                    amplitude = squeeze(allAmplitudes(c1,c2,:));
                    phase = squeeze(allPhases(c1,c2,:));
                    
                    if  c1 == c2
                        subplot(ha(c1+C*(c2-1)))
                        if opts.logAmplitude
                            h = semilogy(s,amplitude,'LineWidth',2);
                        else
                            h = plot(s,amplitude,'LineWidth',2);
                        end
                        ylim([opts.minAmplitude,opts.maxAmplitude])
                        xlim([opts.minFreq,opts.maxFreq])
                        
                        text(.95*opts.maxFreq,.95*opts.maxAmplitude,...
                            opts.names{c1},...
                            'FontSize',14, ...
                            'FontName','Ubuntu', ...
                            'FontWeight','bold', ...
                            'HorizontalAlignment','right', ...
                            'VerticalAlignment','top', ...
                            'Interpreter','latex');
                        
                        if c1 == 1
                            if opts.showAmplitude
                                ax = gca;
                                GP.spectrumPlots.makeAmplitudeLabels(ax,opts,...
                                                                get(h,'color'))
                            else set(gca,'YTick',[]);
                            end
                            ylabel('Amplitude','FontSize',12,'color',...
                                    get(h,'color'))
                        else set(gca,'YTick',[]);
                        end
                        
                        xlabel('$f$, Hz','FontSize',12,'interpreter','latex')
                        set(gca,'TickLength',[0 0]);
                        
                        GP.spectrumPlots.plotVertLines(opts.vertLines);
                    else
                        subplot(ha(c1+C*(c2-1)));
                        ax = GP.spectrumPlots.plotOneCsd(opts,amplitude,phase);
                                                
                        set(ax(1),'YTick',[]);
                        set(ax(1),'XTick',[]); set(ax(2),'XTick',[]);
                        
                        if c1 == C && c2 == 1
                            set(ax(2),'YTick', linspace(-pi+.2,pi-.2,5), ...
                                'YTickLabel',fix(100*linspace(-pi,pi,5))/100, ...
                                'TickLength',[0 0]);
                            ylabel(ax(2),'Phase','FontSize',12);
                        else set(ax(2),'YTick',[]);
                        end
                    end
                end
            end
            drawnow;
        end
        
        function ax = plotCsdComp(s,UKU,varargin)
            % PLOTCSDCOMP   Plot the cross-spectral density of a single
            % cross-spectral component.
            %    PLOTCSDCOMP(s,UKU) plots the cross spectrum component at
            %    the frequency locations indicated in the N-dimensional
            %    vector s.  The input UKU is a C-by-C-by-N array, holding
            %    the block diagonal elements of the discrete Fourier
            %    transform of the covariance matrix, U*K*U', where U is the
            %    unitary DFT matrix.
            %
            %    PLOTCSDCOMP(s,UKU,'param1',value1,'param2',value2,...)
            %    plots the cross spectrum domponent with the specified
            %    parameters.  Note that the param-value pairs may be
            %    included in any order.
            %
            %    Parameters:
            %       minFreq         - Minimum frequency (default 0)
            %       maxFreq         - Maximum frequency (default 30)
            %       logAmplitude    - Plot in log-space (default false)
            %       minAmplitude    - Minimum amplitude (default 1e-4)
            %       chan1           - First channel number (default 1)
            %       chan2           - Second channel number (default 2)
            %       styles          - Cell array of plot styles for
            %                         cross-amplitude/phase spectra
            %                         (default {'-','-'})
            %       colors          - 2-by-3 matrix of plot colors for
            %                         cross-amplitude/phase spectra
            %                         (default blue/red)
            %       vertLines       - Array of frequency locations to draw
            %                         dashed vertical lines (default [4 8
            %                         13 30] to reflect brain wave bands)
            %       ax              - Provided axis handle
            %
            %    Output:
            %       ax - axis handle for future use
            %
            %    Examples:
            %       figure; ax = PLOTCSDCOMP(s,UKU);
            %       PLOTCSDCOMP(s,UKU2,'ax',ax,'styles',{'--','--'});
            %    will plot the cross-spectrum between channels 1 and 2 of
            %    both UKU (solid lines) and UKU2 (dashed lines) on the same
            %    plot.
            
            % parse variable number of input arguments into opts
            p = GP.spectrumPlots.csdInputParser();
            addParameter(p,'chan1',1,@isnumeric);
            addParameter(p,'chan2',2,@isnumeric);
            parse(p,s,UKU,varargin{:});
            opts = p.Results;
            
            allAmplitudes = abs(UKU);
            allPhases = wrapToPi(angle(UKU));
            opts.maxAmplitude = max(allAmplitudes(:));
            opts.minAmplitude = GP.spectrumPlots.calculateMinAmplitude(opts,...
                allAmplitudes);
            
            amplitude = squeeze(allAmplitudes(opts.chan1,opts.chan2,:));
            phase = squeeze(allPhases(opts.chan1,opts.chan2,:));
            
            ax = GP.spectrumPlots.plotOneCsd(opts,amplitude,phase);
            
            GP.spectrumPlots.makeAmplitudeLabels(ax(1),opts,opts.colors(1,:))
            
            ylabel(ax(1),'Amplitude','FontSize',11,'Color',opts.colors(1,:))
            
            set(ax(2),'YTick', linspace(-pi+.2,pi-.2,5), ...
                'YTickLabel',fix(100*linspace(-pi,pi,5))/100, ...
                'TickLength',[0 0],'YColor',opts.colors(2,:));
            ylabel(ax(2),'Phase','FontSize',11,'Color',opts.colors(2,:));
            
            xlabel(ax(1),'Frequency','FontSize',11)
            set(ax(1),'TickLength',[0 0]);
            set(ax(2),'TickLength',[0 0]);
        end
        
    end
    
    methods(Static, Access = private)
        
        function parser = csdInputParser()
            % parser - inputParser with parameters common to all plot functions
            
            parser = inputParser;
            p.KeepUnmatched = true;
            addRequired(parser,'s',@isnumeric)
            addRequired(parser,'UKU',@isnumeric)
            addParameter(parser,'minFreq',0,@isnumeric)
            addParameter(parser,'maxFreq',30,@isnumeric)
            addParameter(parser,'logAmplitude',false,@islogical)
            % minAmplitude is only used if logAmplitude is true
            addParameter(parser,'minAmplitude',0,@isnumeric)
            addParameter(parser,'maxAmplitude',nan,@isnumeric)
            addParameter(parser,'vertLines',[4 8 13 30],@isnumeric)
            addParameter(parser,'zeroLine',true,@islogical)
            addParameter(parser,'ax',[])
            addParameter(parser,'styles',{'-','-'},@iscell)
            colors = get(groot,'DefaultAxesColorOrder');
            addParameter(parser,'colors',colors([1,2],:),@isnumeric)
        end
        
        function minAmplitude = calculateMinAmplitude(opts, allAmplitudes)
            % minAmplitude: lower limit for plot y scale. Returns the
            %   minimum amplitude among all PSDs (not CSDs) unless the
            %   pre-specified minAmplitude parameter from the 'main' function is
            %   larger. Inthat case that case, that value is returned
            %
            %   Parameter:
            %       opts - structure containing plot input parameters
            %   Output:
            %       minAmplitude - scalar lower bound on y scale for PSD plot
            %   Example:
            %       opts.minAmplitude = ...
            %        GP.spectrumPlots.calculateMinAmplitude(opts,allAmplitudes);
            
            C = size(allAmplitudes,1);
            n = size(allAmplitudes,3); % num frequency points in psd
            
            % get amplitudes for only PSDs (not CSDs)
            psdAmplitudes = zeros(C,size(allAmplitudes,3));
            for i = 1:C
                psdAmplitudes(1+(i-1)*n:i*n) =  allAmplitudes(i,i,:);
            end
            
            % find min of psdAmplitudes and specified limit
            minAmplitude = min(psdAmplitudes(:));
            minAmplitude = max(minAmplitude, opts.minAmplitude);
            if minAmplitude > opts.maxAmplitude
                error('specified minAmplitude is too large')
            end
        end
        
        function ax = plotOneCsd(opts, amplitude, phase)
            % plotOneCsd: plot a single cross-spectral density between 2 areas
            %
            %    Parameters:
            %       amplitude  - vector containing CSD amplitudes to plot
            %       phase      - vector containing CSD phases to plot
            %       opts.ax    - normally empty. to overlay one CSD on
            %                    another, ax must be an array of the 2 axes
            %                    handles in the original CSD plot
            %       (see above for descriptions of other params in 'opts')
            %
            %   Output:
            %      ax - axis handle for future use
            %   Example:
            %       ax = GP.spectrumPlots.plotOneCsd(opts,amplitude,phase);
            
            s = opts.s;
            % replace phase in discontinuity function with NaNs
            [sp,phase] = util.removeDiscontinuities(s,phase);
            
            ax = opts.ax;
            if ~isempty(ax)
                hold(ax(1),'on');
                plot(ax(1),s,amplitude,'LineStyle', ...
                    opts.styles{1},'Color',opts.colors(1,:), ...
                    'LineWidth',2);
                hold(ax(1),'off');
                
                hold(ax(2),'on');
                plot(ax(2),sp,phase,'LineStyle', ...
                    opts.styles{2},'Color',opts.colors(2,:), ...
                    'LineWidth',2);
                if opts.zeroLine
                    plot(ax(2),sp,zeros(size(sp)),'r:')
                end
                hold(ax(2),'off');
            else
                if opts.logAmplitude, fn1 = 'semilogy';
                else fn1 = 'plot';
                end
                [ax,h1,h2] = plotyy(s,amplitude,sp,phase,fn1,'plot');
                
                hold(ax(2),'on');
                if opts.zeroLine
                    plot(ax(2),sp,zeros(size(sp)),'r:')
                end
                hold(ax(2),'off');
                
                set(h1,'LineWidth',2,'LineStyle',opts.styles{1}, ...
                    'Color',opts.colors(1,:));
                set(h2,'LineWidth',2,'LineStyle',opts.styles{2}, ...
                    'Color',opts.colors(2,:));
                ylim(ax(2),[-pi,pi])
                ylim(ax(1),[opts.minAmplitude,opts.maxAmplitude])
                xlim(ax(1),[opts.minFreq,opts.maxFreq])
                xlim(ax(2),[opts.minFreq,opts.maxFreq])
                
                GP.spectrumPlots.plotVertLines(opts.vertLines);
            end
        end
        
        function plotVertLines(vLines)
            % plotVertLines   Plot vertical lines, dividing regions of a
            % spectral density plot
            %
            %    Parameter:
            %       vLines - array of frequecies at which to plot lines
            %
            %    Example:
            %       GP.spectrumPlots.plotVertLines(opts.vertLines);
            
            for loc = vLines
                hold(gca,'on');
                ylims = get(gca,'ylim');
                plot([loc loc],ylims,'k:')
                hold(gca,'off');
            end
        end
        
        function makeAmplitudeLabels(ax,opts,color)
            
            % check if plot has log y-scale, adjust tick labels accordingly
            if opts.logAmplitude || strcmp(ax.YScale,'log')
                logBottom = 0.95*log10(opts.minAmplitude)+...
                    0.05*log10(opts.maxAmplitude);
                logTop = 0.05*log10(opts.minAmplitude)+...
                    0.95*log10(opts.maxAmplitude);
                yTick = logspace(logBottom,logTop,3);
            else
                bottom = 0.95*opts.minAmplitude+...
                    0.05*opts.maxAmplitude;
                top = 0.05*opts.minAmplitude+...
                    0.95*opts.maxAmplitude;
                yTick = linspace(bottom,top,3);
            end
            yTickLabels = fix(100*yTick)/100;
            
            set(ax,'YTick',yTick,'YTickLabel',yTickLabels,...
                'TickLength',[0 0],'ycolor',color);
        end
        
    end
end