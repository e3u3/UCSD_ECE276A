function output = control_update( input,control_input )
output = zeros(size(input));
    if(control_input==0)
        output = input;
    elseif(control_input==1)
        for z=3:29
        %if the control input is 1, there are only three possible start points(z-2,z-1,z)
            output(z)=input(z-2)*1/3+input(z-1)*1/2+input(z)*1/6; 
        end
        output(27)=output(27)+output(28)+output(29); %if it exceeds 25, it would be reset to 25
        output(28)=0;output(29)=0;
    elseif(control_input==-1)
        for z=1:27
        %if the control input is -1, there are only three possible start points(z,z+1,z+2)
            output(z)=input(z+2)*1/3+input(z+1)*1/2+input(z)*1/6; 
        end
        output(3)=output(1)+output(2)+output(3); %if it goes below 1, it would be reset to 1
        output(1)=0;output(2)=0;
    elseif(control_input==2)
        for z=3:30
        %if the control input is 2, there are only three possible start points(z-3,z-2,z-1)    
            if(z==3)
                output(z)=0*1/3+input(z-2)*1/2+input(z-1)*1/6; 
            else
                output(z)=input(z-3)*1/3+input(z-2)*1/2+input(z-1)*1/6;
            end
        end
        output(27)=output(27)+output(28)+output(29)+output(30); %if it exceeds 25, it would be reset to 25
        output(28)=0;output(29)=0;output(30)=0;
    end
end

