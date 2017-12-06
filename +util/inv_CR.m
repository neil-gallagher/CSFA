    function Z = inv_CR(M)
    %     Complex matrix inversion via Real matrix inversion
    %     F C Chang     01/28/15

             A = real(M);
             B = imag(M);
             X = inv(A+B*inv(A)*B);
             Y = inv(B+A*inv(B)*A);
             Z = complex(X,-Y);