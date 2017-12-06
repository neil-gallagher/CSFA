function AnalyticSignal = analyticSignal(x)
% analyticSignal
% -----------------
% x: input data
% AnalyticSignal: structure containing phase and amplitude of analytic
% component of 'theLFP'

    hilbertX = hilbert(x);
    hilbertPhaseArray = atan2 (imag (hilbertX), real (hilbertX));
    hilbertAmplitudeArray = sqrt((real(hilbertX).^2+imag (hilbertX).^2)');

    AnalyticSignal.phase(1:length(hilbertPhaseArray)) = hilbertPhaseArray;
    AnalyticSignal.amplitude = hilbertAmplitudeArray;

end

