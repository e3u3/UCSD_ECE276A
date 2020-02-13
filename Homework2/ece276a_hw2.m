P_mass_function=zeros(11,30); %padding for the convevience of computation
P_mass_function(1,3:27)=1/25;
door_location=[10,15,19];%because I add 2 blank cell in the head of the list, the position of the door would also change.
u=[1,-1,2,1,1];
z=[1,-1,-1,-1,1];
subplot(3,4,1);
bar(P_mass_function(1,3:27));
ylim([0,0.5])
index=1;
title(mat2str(index));
for i=1:5 %total 5 times for each pair of observation and control input
    for j=1:2 %Assume the robot moves before observation
        if(j==1) %prediction
            output = control_update(P_mass_function(1+2*(i-1),:),u(i));
            P_mass_function(1+2*(i-1)+j,:) = output;
            index=index+1;
            subplot(3,4,index);
            bar(output(1,3:27));
            ylim([0,0.5])
            title(mat2str(index));
        else     %correction from sensor data
            output = correction(P_mass_function(1+2*(i-1)+1,:),door_location,z(i));
            P_mass_function(1+2*(i-1)+j,:) = output;
            index=index+1;
            subplot(3,4,index);
            bar(output(1,3:27));
            ylim([0,0.5])
            title(mat2str(index));
        end
    end
end
P_mass_function=P_mass_function(:,3:27);
P_mass_function=P_mass_function+0.0000000000001;
P=zeros(5,25);   %4*25 P(x_0|z_{0:4},u_{0:4}) ~ P(x_4|z_{0:4},u_{0:4})
P(5,:)=P_mass_function(11,:);
for i=4:-1:1
        move_step=u(i+1);
        for z=1:25
            if(move_step==1)
                if(z==1)
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z)/P_mass_function(11-(5-i)*2,z));
                elseif(z==2)
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z)/P_mass_function(11-(5-i)*2,z)+(1/2)*P(1+i,z-1)/P_mass_function(11-(5-i)*2,z-1));
                else
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z)/P_mass_function(11-(5-i)*2,z)+(1/2)*P(1+i,z-1)/P_mass_function(11-(5-i)*2,z-1)+(1/3)*P(1+i,z-2)/P_mass_function(11-(5-i)*2,z-2));
                end
            elseif(move_step==-1)
                if(z==25)
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z)/P_mass_function(11-(5-i)*2,z));
                elseif(z==24)
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z)/P_mass_function(11-(5-i)*2,z)+(1/2)*P(1+i,z+1)/P_mass_function(11-(5-i)*2,z+1));
                else
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z)/P_mass_function(11-(5-i)*2,z)+(1/2)*P(1+i,z+1)/P_mass_function(11-(5-i)*2,z+1)+(1/3)*P(1+i,z+2)/P_mass_function(11-(5-i)*2,z+2));
                end
            elseif(move_step==2)
                if(z==1)
                    P(i,z)=0;
                elseif(z==2)
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z-1)/P_mass_function(11-(5-i)*2,z-1));
                elseif(z==3)
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z-1)/P_mass_function(11-(5-i)*2,z-1)+(1/2)*P(1+i,z-2)/P_mass_function(11-(5-i)*2,z-2));
                else
                    P(i,z)=P_mass_function(11-(6-i)*2+1,z)*((1/6)*P(1+i,z-1)/P_mass_function(11-(5-i)*2,z-1)+(1/2)*P(1+i,z-2)/P_mass_function(11-(5-i)*2,z-2)+(1/3)*P(1+i,z-3)/P_mass_function(11-(5-i)*2,z-3));
                end
            end
        end
    P(i,:)=P(i,:)/sum(P(i,:));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_initial=zeros(1,25); %from the second last step to the last one.
%This is necessary beacuse in my codes, control update happens before observation.
move_step=u(1);
i=0;
 for z=1:25
      if(move_step==1)
           if(z==25)
                P_initial(1,z)=(1/6)*P(1+i,z);
           elseif(z==24)
                P_initial(1,z)=(1/6)*P(1+i,z)+(1/2)*P(1+i,z+1);
           else
                P_initial(1,z)=(1/6)*P(1+i,z)+(1/2)*P(1+i,z+1)+(1/3)*P(1+i,z+2);
           end
      end
 end
P_initial(1,:)=P_initial(1,:)/sum(P_initial(1,:));