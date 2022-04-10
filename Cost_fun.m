x = [1 2 3];
a = -0.5;
y = x;
subplot(1,3,1)
b = 0;
h = a*x+b;
J = sum((h-y).^2/(2*length(x)));
plot(a,J,'*','markersize',5)
hold on
% subplot(1,3,2)
% for b = -2:0.5:4
%     a = 1;
%     h = a*x+b;
%     J = sum((h-y).^2/(2*length(x)));
%     plot(b,J,'*','markersize',5)
%     hold on
% end
% subplot(1,3,3)
% for a = -2:0.5:4
%     h = a*x+a;
%     J = sum((h-y).^2/(2*length(x)));
%     plot(a,J,'*','markersize',5)
%     hold on
% end
