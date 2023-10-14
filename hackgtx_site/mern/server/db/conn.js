const { MongoClient } = require("mongodb");
const con_str = process.env.ATLAS_URI;
const client = new MongoClient(con_str);

let conn;
let _db;

module.exports = {
  connectToServer: async function (callback) {

    try {
      conn = await client.connect();
    } catch (e) {
      console.error(e);
    }

    _db = conn.db("employees");
    console.log("Connection to MongoDB was successful.");

    return (_db === undefined ? false : true);
  },
  getDb: function () {
    return _db;
  },
};