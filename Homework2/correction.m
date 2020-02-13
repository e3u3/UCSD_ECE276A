function output = correction( input,door_location,measurement )
output=zeros(size(input));
    for i=3:27
        if(measurement==-1)
            output(i)=input(i)*0.8;
        else
            output(i)=input(i)*0.2;
        end
    end
    if(measurement==-1)
        for j=1:3
           output(door_location(j))=output(door_location(j))/0.8*0.1;
           output(door_location(j)+1)=output(door_location(j)+1)/0.8*0.3;
           output(door_location(j)-1)=output(door_location(j)-1)/0.8*0.3;
        end
    else
        for j=1:3
           output(door_location(j))=output(door_location(j))/0.2*0.9;
           output(door_location(j)+1)=output(door_location(j)+1)/0.2*0.7;
           output(door_location(j)-1)=output(door_location(j)-1)/0.2*0.7;
        end 
    end
    pro=sum(output); %normalize to 1
    output=output/pro;
end

