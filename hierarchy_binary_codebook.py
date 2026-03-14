function  [w_hierarchy_tra] = hierarchy_binary_codebook(Nt)
    l=log2(Nt);
    w_hierarchy_tra=cell(l,1);
    for i=1:l
        w_hierarchy_tra{i}=zeros(2^i,Nt);
        for j=1:2^i
            index=((j-1)*Nt/(2^i)+1:j*Nt/(2^i));
            w_hierarchy_tra{i}(j,:)=generate_widebeam(Nt,index);
        end
    end
end