# Run with:
# export FLASK_APP=server.py
# flask run --host=0.0.0.0

from flask import Flask, jsonify, request
import MySQLdb

db  = MySQLdb.connect("localhost", "root", "", "dgmmn")
app = Flask(__name__)

fields = [
  'id', 'href', 'author', 'title', 'year', 'style',
  'processed', 'bars', 'pages', 'requested', 'score_id', 'instrumentation'
]


def to_dict(row):
  return { fields[idx]: val for idx, val in enumerate(row) }


@app.route("/poll")
def poll():
  cursor = db.cursor()

  try:
    cursor.execute("SELECT * FROM scores WHERE processed = 0 AND requested = 0 ORDER BY id LIMIT 1")
    row = to_dict(cursor.fetchone())

    cursor.execute("UPDATE scores SET requested = 1 WHERE id = %d" % row['id'])
    db.commit()

    return jsonify(**row)
  except Exception as e:
    print("Error: %s" % str(e))
    db.rollback()

  return jsonify(None)


@app.route("/complete/<int:score_id>")
def complete(score_id):
  bars    = int(request.args.get('bars'))
  pages   = int(request.args.get('pages'))
  scanned = int(request.args.get('scanned'))

  cursor = db.cursor()
  try:
    cursor.execute("""
      UPDATE scores
      SET processed = 1, bars = %d, pages = %d, scanned = %d
      WHERE score_id = %d
    """ % (bars, pages, scanned, score_id))
    db.commit()

    return jsonify(True)
  except Exception as e:
    print("Error: %s" % str(e))

  return jsonify(False)


@app.route("/info/<int:score_id>")
def info(score_id):
  is_scanned = int(request.args.get('scanned'))

  cursor = db.cursor()
  try:
    cursor.execute(
      "UPDATE scores SET scanned = %d WHERE score_id = %d" % (is_scanned, score_id)
    )
    db.commit()
    return jsonify(True)
  except Exception as e:
    print("Error: %s" % str(e))

  return jsonify(False)

if __name__ == "__main__":
  app.run()
