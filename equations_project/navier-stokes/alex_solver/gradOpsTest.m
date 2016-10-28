function gradOpsTest()
    f = peaks();
    gOps = GradOps2D(size(f,2), size(f,1), 1, 1);
    test_grad(f, gOps)
    test_v_dot_grad(f, gOps)
    test_dot_grad_v({f,f.^2}, gOps)
end

function test_grad(f, gOps)
    figure(1)
    g = gOps.apply_grad(f);
    subplot(2,1,1)
    imagesc([g{1}, g{2}]); axis image
    subplot(2,1,2)
    [gx,gy] = gradient(f);
    imagesc([gx, gy]); axis image
end

function test_v_dot_grad(f, gOps)
    figure(2)
    v = {cos(f),sin(f)};
    g = gOps.apply_v_dot_grad(f, v);
    subplot(2,1,1)
    imagesc(g); axis image
    subplot(2,1,2)
    [gx,gy] = gradient(f);
    imagesc(v{1}.* gx + v{2} .* gy); axis image
end

function test_dot_grad_v(f, gOps)
    figure(3)
    v = {cos(f{1}),sin(f{1})};
    g = gOps.apply_dot_grad_v(f, v);
    subplot(2,1,1)
    imagesc([g{1}, g{2}]); axis image
    subplot(2,1,2)
    [gx1,gy1] = gradient(v{1});
    [gx2,gy2] = gradient(v{2});
    dgv1 = f{1}.*gx1 + f{2}.*gy1;
    dgv2 = f{1}.*gx2 + f{2}.*gy2;
    imagesc([dgv1, dgv2]); axis image
end