# laymau

## usage

The app need an `postgres` sql server, and a AWS'S3 competible storage service, such as `minio`.
If you don't have any yet, the `db` directory might be a good start.

run the app using

- activate the production mode rather than the default dev mode, by specify a eviroment variable `PRODUCTION` `export PRODUCTION=1`
- first, 3 reference images are needed for each person.
Use `p` to pause video, use mouse to select ROI (ROI must be contained only one face, which should be a ref face), `Enter` after select roi. Repeat the process 2 time to get all 3 ref image.
- select a ROI for laymau
- (after at least 4 times of selection roi) the app automatically do detection the entire frame, and save all faces. Each 100 faces detection, the app will report number of good face like this:
    > current status: #frontal: 72  #threeforth: 16 #sideview: 2    #other: 7

    once `forntal > 70 and threforth > 70 and sideview > 10` app will stop automatically

```bash
python3 app/app.py collect process test/test.mp4 datnt527 --num_facehandler 3
```

## development

- when developing, it's critical to use a seperated database and storage service rather than use production ones. To create your own database and storage for development, use `docker-compose.yml` in `db` directory, then modify settings in `app/script/config.py`
- By default, app is run in development mode (use seperated, dev-only database and storage)

**Note:**
In dev-mode, to reset the database:

```bash
# remove the db
export PGPASSWORD="123456" && psql -U postgres -h localhost -c "DROP DATABASE IF EXISTS laymau;" && psql -U postgres -h localhost -c "CREATE DATABASE laymau;"  && psql -U postgres -h localhost -d laymau -a -f db/postgresdb.sql
# remove MinIO bucket as well
mc rb laymau --force
```
