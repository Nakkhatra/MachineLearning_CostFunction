function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

[row n]=size(X);
n=n-1;

J=0;

for iter = 1:num_iters
    pred=X*theta;
    sum_devs=zeros(n+1,1);
    for i=1:m
        deviation=pred(i)-y(i);
        xvector=(X(i,:))';
        deviation=deviation*xvector;
        A=X(i,:);
        J_add=(((theta)'*A')-y(i)).^2;
        J=J+J_add;
        sum_devs=sum_devs+deviation;
    end;
    
    delta=sum_devs*(1/m);
    theta=theta-alpha*delta;
    J=(1/(2*m))*J;
    J_history(iter)=J;
    
end;

iter=1:1:num_iters;
plot(iter',J_history)
    xlabel('No. of iterations')
    ylabel('Cost Function')
    title('Cost function vs iterations')
    print -dpng 'Costfunction_vs_iterationfinal_multivariate.png'

theta
J
      










    % ============================================================

    % Save the cost J in every iteration   


end
