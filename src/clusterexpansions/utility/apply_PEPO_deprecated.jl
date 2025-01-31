function own_isometry(Ws)
    for dir in 1:4
        (D, DO) = dims(Ws[dir].codom)
        for k in 1:DO
            if dir == 1
                Oᵏ = sum(O[][:, :, k, :, :, :])
            elseif dir == 2
                Oᵏ = sum(O[][:, :, :, k, :, :])
            elseif dir == 3
                Oᵏ = sum(O[][:, :, :, :, k, :])
            elseif dir == 4
                Oᵏ = sum(O[][:, :, :, :, :, k])
            end
            for j in 1:D
                Ws[dir][j, k, j] = 1.0 / (Oᵏ * DO)
            end
        end
    end
    return Ws
end

