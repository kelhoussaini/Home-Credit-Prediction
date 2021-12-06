# create the app in the EU region
heroku create homecredit --ssh-git --region eu

# you should see the origin remote for GitHub
# and the heroku remote for Heroku
git remote -v

# manually add the heroku remote if necessary
git remote add heroku https://git.heroku.com/homecredit.git

# deploy the app
git push heroku master

# start the web dyno
heroku ps:scale web=1

# check the logs for errors
heroku logs --tail