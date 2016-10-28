classdef GradOps2D < handle
    properties
       x_dim
       y_dim
       n_xy
       dx
       dy
       
       grad
    end
    
    methods
        function obj = GradOps2D(x_dim, y_dim, dx, dy)
            obj.x_dim = x_dim;
            obj.y_dim = y_dim;
            obj.dx = dx;
            obj.dy = dy;
            obj.n_xy = x_dim * y_dim;
        end

        function idx = ravel(self, x, y)
            idx = x + (y - 1) * self.x_dim;
        end

        function grad = make_grad(self)
       
            function grad_y = gradient_y()
                grad_y = sparse(self.n_xy, self.n_xy);
                for x = 1:self.x_dim
                    % central differences in bulk
                    for y = 2:(self.y_dim - 1)
                        fwd_idx = self.ravel(y + 1, x);
                        bkwd_idx = self.ravel(y - 1, x);
                        out_idx = self.ravel(y, x);
                        grad_y(out_idx, fwd_idx) = 1;
                        grad_y(out_idx, bkwd_idx) = -1;
                    end

                    % forward differences at y = 0
                    fwd_idx = self.ravel(2, x);
                    out_idx = self.ravel(1, x);
                    grad_y(out_idx, fwd_idx) = 2;
                    grad_y(out_idx, out_idx) = -2;

                    % backward differences at y = self.y_dim
                    bkwd_idx = self.ravel(self.y_dim - 1, x);
                    out_idx = self.ravel(self.y_dim, x);
                    grad_y(out_idx, out_idx) = 2;
                    grad_y(out_idx, bkwd_idx) = -2;
                end
                
                grad_y = grad_y * 0.5 * self.dy;
            end
            
            function grad_x = gradient_x()
                grad_x = sparse(self.n_xy, self.n_xy);
                
                for y = 1:self.y_dim
                    % central differences in bulk
                    for x = 2:(self.x_dim - 1)
                        fwd_idx = self.ravel(y, x + 1);
                        bkwd_idx = self.ravel(y, x - 1);
                        out_idx = self.ravel(y, x);
                        grad_x(out_idx, fwd_idx) = 1;
                        grad_x(out_idx, bkwd_idx) = -1;
                    end
                    
                    % forward differences at x = 0
                    fwd_idx = self.ravel(y, 2);
                    out_idx = self.ravel(y, 1);
                    grad_x(out_idx, fwd_idx) = 2;
                    grad_x(out_idx, out_idx) = -2;

                    % backward differences at x = self.x_dim
                    bkwd_idx = self.ravel(y, self.x_dim - 1);
                    out_idx = self.ravel(y, self.x_dim);
                    grad_x(out_idx, out_idx) = 2;
                    grad_x(out_idx, bkwd_idx) = -2;
                end
                
                grad_x = grad_x * 0.5 * self.dx;
            end
            
            grad = {gradient_x(), gradient_y()};
            self.grad = grad;
        end
        
        function v_dot_grad = make_v_dot_grad(self, v)
            v_dot_grad = sparse(self.n_xy, self.n_xy);
            if isempty(self.grad) self.make_grad(); end
            for i = [1,2]
                vi = v{i};
                rep_v = repmat(vi(:), [1, self.n_xy]);
                v_dot_grad = v_dot_grad + rep_v .* self.grad{i};
            end
        end
        
        function dot_grad_v = make_dot_grad_v(self, v)
            dot_grad_v = cell(2,2);
            if isempty(self.grad) self.make_grad(); end
            for i = [1,2]
                for j = [1,2]
                    dot_grad_v{i,j} = diag(self.grad{i} * v{j}(:));
                end
            end
        end
        
        function g = apply_grad(self, f)
            if isempty(self.grad) self.make_grad(); end
            tf = f;
            g_x = self.grad{1} * tf(:);
            g_y = self.grad{2} * tf(:);
            g = {reshape(g_x, size(f)), reshape(g_y, size(f))};
        end
        
        function g = apply_v_dot_grad(self, f, v)
            v_dot_grad = self.make_v_dot_grad(v);
            g = reshape(v_dot_grad * f(:), size(f));
        end
        
        function g = apply_dot_grad_v(self, f, v)
            dot_grad_v = self.make_dot_grad_v(v);
            g_x = dot_grad_v{1,1} * f{1}(:) + dot_grad_v{2,1} * f{2}(:);
            g_y = dot_grad_v{1,2} * f{1}(:) + dot_grad_v{2,2} * f{2}(:);
            g = {reshape(g_x, size(f{1})), reshape(g_y, size(f{1}))};
        end
    end
end