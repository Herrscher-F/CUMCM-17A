function result = solve_quadratic(a, b, c)
    % 求解二次方程 ax^2 + bx + c = 0
    % 输入：系数 a, b, c
    % 输出：方程的解
    
    discriminant = b^2 - 4*a*c;
    
    if discriminant > 0
        x1 = (-b + sqrt(discriminant)) / (2*a);
        x2 = (-b - sqrt(discriminant)) / (2*a);
        result = [x1, x2];
        fprintf('方程有两个实数解：x1 = %.4f, x2 = %.4f\n', x1, x2);
    elseif discriminant == 0
        x = -b / (2*a);
        result = x;
        fprintf('方程有一个重根：x = %.4f\n', x);
    else
        real_part = -b / (2*a);
        imag_part = sqrt(-discriminant) / (2*a);
        fprintf('方程有两个复数解：\n');
        fprintf('x1 = %.4f + %.4fi\n', real_part, imag_part);
        fprintf('x2 = %.4f - %.4fi\n', real_part, imag_part);
        result = [complex(real_part, imag_part), complex(real_part, -imag_part)];
    end
end