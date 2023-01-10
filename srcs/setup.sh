# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    setup.sh                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kkim <kkim@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/01/10 12:21:49 by kkim              #+#    #+#              #
#    Updated: 2023/01/10 12:37:58 by kkim             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

python3 -m pip install --upgrade pip
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv

python3 -m virtualenv ft_env #create env
source ft_env/bin/activate   #activate env

pip install pandas
pip install numpy
pip install matplotlib

which python
