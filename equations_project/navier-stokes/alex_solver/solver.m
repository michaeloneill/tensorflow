% try to solve v.grad(v) = f iteratively
% initial guess: v0
% correction: dv.grad(v0) + v0.grad(dv) = f - v0.grad(v0)

function v0 = solver()
    x_dim = 20;
    y_dim = 20;
    
    gOps = GradOps2D(x_dim, y_dim, 1, 1);
    
    [x,y] = meshgrid(1:x_dim, 1:y_dim);
    %f = {0.1 * x + 0*0.1 * y, 0.1*y};
    %f = {0 * x, 0.005*y-0.005*x};
    f1 = 0.002*(y-y_dim/2)*exp(-(x/2).^2/(2*1^2)-(y-y_dim/2).^2/(2*5^2));
    f2 = -0.002*(y-y_dim/2)*exp(-(x-10/2).^2/(2*1^2)-(y-y_dim/2).^2/(2*5^2));
    f = {-0.001*x,f1+f2+0.001*y};
    
    v0 = make_initial_guess(x_dim, y_dim);
    
    obs = [(1:y_dim)',1*ones(y_dim,1)];
    %obs = [obs;5,5];
    for i = 1:5000
        disp(i)
        b = make_RHS(f, v0, gOps);
        A = make_LHS(v0, gOps);
        A_bc = fix_observations(A, obs, gOps);

        soln = A_bc \ b;
        soln = reintroduce_observations(soln, obs, gOps);
        dv = reshape_solution(soln, gOps);
        delta = 0.01;
        v0 = {v0{1} + delta * dv{1}, v0{2} + delta * dv{2}};
        
        %imagesc([v0{1}, v0{2}]); axis image;
        quiver(x,y,v0{1},v0{2}); axis image;
        
        drawnow();
        %pause(1)
    end
    
end

function init = make_initial_guess(x_dim, y_dim)
    init_x = ones(y_dim, x_dim);
    init_y = zeros(y_dim, x_dim);
    init = {init_x, init_y};
end

function RHS = make_RHS(f, v0, gOps)
    %f - v0.grad(v0)
    [v01x, v01y] = gradient(v0{1});
    [v02x, v02y] = gradient(v0{2});
    rhs1 = f{1} - v0{1}.*v01x + v0{2}.*v01y;
    rhs2 = f{2} - v0{1}.*v02x + v0{2}.*v02y;
    RHS = [rhs1(:); rhs2(:)];
end

function LHS = make_LHS(v0, gOps)
    %dv.grad(v0) + v0.grad(dv)
    dgv = gOps.make_dot_grad_v(v0);
    vdg = gOps.make_v_dot_grad(v0);
    LHS = [dgv{1,1} + vdg, dgv{2,1}; dgv{1,2}, dgv{2,2} + vdg];
end

function A = fix_observations(A, obs, gOps)
    idx = gOps.ravel(obs(:,1), obs(:,2));
    fixed_idx = [idx, idx + size(A,2)/2];
    free_idx = setdiff(1:size(A,2), fixed_idx);
    A = A(:, free_idx);
end

function s = reintroduce_observations(soln, obs, gOps)
    idx = gOps.ravel(obs(:,1), obs(:,2));
    fixed_idx = [idx, idx + gOps.n_xy];
    s_mask = ones(2*gOps.n_xy, 1);
    s_mask(fixed_idx) = 0;
    s = zeros(2*gOps.n_xy, 1);
    s(logical(s_mask)) = soln;
end

function soln = reshape_solution(soln, gOps)
    soln_x = soln(1:gOps.n_xy);
    soln_y = soln((gOps.n_xy+1):end);
    soln = {reshape(soln_x, [gOps.x_dim, gOps.y_dim]), reshape(soln_y, [gOps.x_dim, gOps.y_dim])};
end