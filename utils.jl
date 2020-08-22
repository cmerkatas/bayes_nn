# helper function
function cell2array(Z)
    m, n = size(Z)
    ZZ = zeros(m, n)
    for i in 1:1:m
        for j in 1:1:n
            ZZ[i,j] = Z[i,j][1]
        end
    end
    return ZZ
end
